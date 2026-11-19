"""Stage 1: supervised pre-training on clean source-domain data (Algorithm 1, lines 1-12).

Learning-rate schedules follow Sec. 4.1: the denoiser uses 1e-4 decayed by 20%
every 100 epochs, the predictor uses 5e-4 with a 50-epoch warm-up followed by a
10% decay every 100 epochs.  Gradients are accumulated to reach the effective
batch size of 16 reported in Appendix C.

Usage
-----
    python -m psgno.train_pretrain --config configs/default.yaml
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from .data_utils import (
    Normalizer,
    SlidingWindowDataset,
    load_auxiliary,
    load_trajectories,
    split_trajectories,
    split_train_val_test,
)
from .losses import denoising_loss, prediction_loss
from .noise import structured_noise
from .utils import (
    build_model,
    count_parameters,
    load_config,
    relative_l2,
    resolve_device,
    save_checkpoint,
    set_seed,
)


def _predictor_schedule(warmup: int, step_size: int, gamma: float):
    """Linear warm-up, then a multiplicative decay every ``step_size`` epochs."""

    def factor(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup
        return gamma ** ((epoch - warmup) // step_size) if step_size > 0 else 1.0

    return factor


def _denoiser_schedule(step_size: int, gamma: float):
    """Multiplicative decay every ``step_size`` epochs, no warm-up."""

    def factor(epoch: int) -> float:
        return gamma ** (epoch // step_size) if step_size > 0 else 1.0

    return factor


def _run_epoch(model, loader, device, cfg, noise_cfg, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    accum = cfg.get("accum_steps", 1)

    totals = {"loss": 0.0, "denoise": 0.0, "predict": 0.0, "rel_den": 0.0, "rel_pred": 0.0}
    seen = 0

    if training:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        if len(batch) == 3:
            x, y, aux = batch
            aux = {key: value.to(device) for key, value in aux.items()}
        else:
            x, y = batch
            aux = None
        x, y = x.to(device), y.to(device)
        noisy, noise, _ = structured_noise(x, **noise_cfg)

        use_amp = cfg["amp"] and device.type == "cuda"
        with torch.autocast("cuda", enabled=use_amp):
            eta = model.denoise(noisy)
            denoised = noisy - eta
            pred = model.predict(denoised, aux)

            loss_d = denoising_loss(eta, noise)
            loss_s = prediction_loss(pred, y)
            loss = cfg["lambda_denoise"] * loss_d + cfg["lambda_predict"] * loss_s

        if training:
            update = (step + 1) % accum == 0 or (step + 1) == len(loader)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss / accum).backward()
                if update:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], cfg["grad_clip"]
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                (loss / accum).backward()
                if update:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], cfg["grad_clip"]
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        batch = x.shape[0]
        seen += batch
        totals["loss"] += loss.item() * batch
        totals["denoise"] += loss_d.item() * batch
        totals["predict"] += loss_s.item() * batch
        totals["rel_den"] += relative_l2(denoised, x).item() * batch
        totals["rel_pred"] += relative_l2(pred, y).item() * batch

    return {key: value / max(seen, 1) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default=None, help="Override the data path.")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data:
        config["data"]["path"] = args.data
    if args.epochs:
        config["pretrain"]["epochs"] = args.epochs

    cfg = config["pretrain"]
    data_cfg = config["data"]
    set_seed(config["seed"])
    device = resolve_device(config["device"])

    # ---- data -------------------------------------------------------------
    raw = load_trajectories(data_cfg["path"])
    source, _ = split_trajectories(raw, data_cfg["source_ratio"])
    train_idx, val_idx, _ = split_train_val_test(source.shape[0])

    normalizer = Normalizer.fit(source[train_idx])
    source = normalizer.transform(source)

    # Physical quantities stored with the data (e.g. the body force of the
    # incompressible case) are normalized with the same statistics whenever
    # they share the channel layout of the state, so that the residual stays
    # dimensionally consistent.
    aux = load_auxiliary(data_cfg["path"], data_cfg.get("aux_keys") or [])
    aux = {
        key: normalizer.transform(value) if value.shape[-1] == raw.shape[-1] else value
        for key, value in aux.items()
    }

    t_in, horizon, stride = data_cfg["t_in"], data_cfg["horizon"], data_cfg["stride"]
    train_set = SlidingWindowDataset(
        source[train_idx], t_in, horizon, stride,
        aux={key: value[train_idx] for key, value in aux.items()},
    )
    val_set = SlidingWindowDataset(
        source[val_idx], t_in, horizon, stride,
        aux={key: value[val_idx] for key, value in aux.items()},
    )

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], drop_last=False)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"])

    # ---- model / optimizer ------------------------------------------------
    model = build_model(config).to(device)
    denoiser_params = list(model.denoiser.parameters())
    denoiser_ids = {id(p) for p in denoiser_params}
    other_params = [p for p in model.parameters() if id(p) not in denoiser_ids]

    optimizer = torch.optim.Adam(
        [
            {"params": denoiser_params, "lr": cfg["lr_denoiser"]},
            {"params": other_params, "lr": cfg["lr_predictor"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            _denoiser_schedule(cfg["step_size"], cfg["lr_denoiser_gamma"]),
            _predictor_schedule(
                cfg["warmup_epochs"], cfg["step_size"], cfg["lr_predictor_gamma"]
            ),
        ],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")

    print(f"device={device}  parameters={count_parameters(model):,}")

    # ---- training loop ----------------------------------------------------
    os.makedirs(cfg["save_dir"], exist_ok=True)
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(cfg["epochs"]):
        train_metrics = _run_epoch(
            model, train_loader, device, cfg, config["noise"], optimizer, scaler
        )
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, device, cfg, config["noise"])
        scheduler.step()

        print(
            f"[{epoch + 1:03d}/{cfg['epochs']}] "
            f"train loss={train_metrics['loss']:.4e} "
            f"den={train_metrics['denoise']:.4e} pred={train_metrics['predict']:.4e} | "
            f"val loss={val_metrics['loss']:.4e} "
            f"rel_den={val_metrics['rel_den']:.4f} rel_pred={val_metrics['rel_pred']:.4f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            bad_epochs = 0
            save_checkpoint(
                os.path.join(cfg["save_dir"], "best.pt"),
                model,
                optimizer,
                normalizer=normalizer.state_dict(),
                epoch=epoch,
                val_loss=best_val,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"early stopping at epoch {epoch + 1}")
                break


if __name__ == "__main__":
    main()
