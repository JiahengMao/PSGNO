"""Stage 2: unsupervised fine-tuning on noisy target-domain data (Algorithm 1, lines 13-28).

Only three parameter groups are updated: the PISG gating strengths
``alpha^(n)``, the physical projections ``W_phys^(n)`` and the LoRA adapters
on the Transformer Q/K/V projections.

Usage
-----
    python -m psgno.train_finetune --config configs/default.yaml
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
from .losses import consistency_loss, physics_loss
from .noise import structured_noise
from .physics import get_residual
from .utils import build_model, load_config, resolve_device, save_checkpoint, set_seed


def _run_epoch(model, loader, device, cfg, noise_cfg, residual_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    accum = cfg.get("accum_steps", 1)

    totals = {"loss": 0.0, "consistency": 0.0, "physics": 0.0}
    seen = 0

    if training:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        if len(batch) == 3:
            buffer, _, aux = batch
            aux = {key: value.to(device) for key, value in aux.items()}
        else:
            buffer, _ = batch
            aux = None
        buffer = buffer.to(device)
        noisy, _, _ = structured_noise(buffer, **noise_cfg)

        with torch.set_grad_enabled(training):
            # Frozen denoiser: no parameters are updated in this stage.
            with torch.no_grad():
                denoised = noisy - model.denoise(noisy)

            history = denoised[:, -cfg["t_in"] :]
            pred = model.predict(history, aux)

            loss_c, _ = consistency_loss(
                model.predict, denoised, t_in=cfg["t_in"],
                horizon=cfg["horizon"], h_c=cfg["h_c"], aux=aux,
            )
            loss_p = physics_loss(residual_fn, pred, aux)
            loss = cfg["lambda_consistency"] * loss_c + cfg["lambda_physics"] * loss_p

            if training:
                update = (step + 1) % accum == 0 or (step + 1) == len(loader)
                (loss / accum).backward()
                if update:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], cfg["grad_clip"]
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        batch = buffer.shape[0]
        seen += batch
        totals["loss"] += loss.item() * batch
        totals["consistency"] += loss_c.item() * batch
        totals["physics"] += loss_p.item() * batch

    return {key: value / max(seen, 1) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default=None, help="Override the target data path.")
    parser.add_argument("--checkpoint", default=None, help="Override the pre-trained checkpoint.")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data:
        config["data"]["path"] = args.data
    if args.epochs:
        config["finetune"]["epochs"] = args.epochs

    cfg = dict(config["finetune"])
    data_cfg = config["data"]
    cfg["t_in"] = data_cfg["t_in"]
    cfg["horizon"] = data_cfg["horizon"]
    set_seed(config["seed"])
    device = resolve_device(config["device"])

    # ---- data -------------------------------------------------------------
    raw = load_trajectories(data_cfg["path"])
    _, target = split_trajectories(raw, data_cfg["source_ratio"])
    train_idx, val_idx, _ = split_train_val_test(target.shape[0])

    checkpoint_path = args.checkpoint or cfg["init_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    normalizer = Normalizer.from_state_dict(checkpoint["normalizer"])
    target = normalizer.transform(target)

    t_in, horizon = data_cfg["t_in"], data_cfg["horizon"]
    h_c = cfg["h_c"]
    buffer_len = t_in + h_c

    # Target-domain auxiliary quantities, normalized with the source-domain
    # statistics exactly as in pre-training.
    cut = int(round(raw.shape[0] * data_cfg["source_ratio"]))
    aux = load_auxiliary(data_cfg["path"], data_cfg.get("aux_keys") or [])
    aux = {
        key: normalizer.transform(value[cut:])
        if value.shape[-1] == raw.shape[-1]
        else value[cut:]
        for key, value in aux.items()
    }

    train_set = SlidingWindowDataset(
        target[train_idx], buffer_len, horizon, data_cfg["stride"],
        aux={key: value[train_idx] for key, value in aux.items()},
    )
    val_set = SlidingWindowDataset(
        target[val_idx], buffer_len, horizon, data_cfg["stride"],
        aux={key: value[val_idx] for key, value in aux.items()},
    )

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"])

    # ---- model ------------------------------------------------------------
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    trainable = model.enable_finetune(cfg["lora_rank"], cfg["lora_alpha"])
    model.to(device)  # move newly created LoRA modules to the device
    print(f"device={device}  trainable parameters={trainable:,}")

    residual_fn = get_residual(
        config["model"].get("residual", "ins"),
        **(config["model"].get("residual_params") or {}),
    )
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["step_size"], gamma=cfg["gamma_lr"]
    )

    os.makedirs(cfg["save_dir"], exist_ok=True)
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(cfg["epochs"]):
        train_metrics = _run_epoch(
            model, train_loader, device, cfg, config["noise"], residual_fn, optimizer
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model, val_loader, device, cfg, config["noise"], residual_fn
            )
        scheduler.step()

        print(
            f"[{epoch + 1:03d}/{cfg['epochs']}] "
            f"train loss={train_metrics['loss']:.4e} "
            f"c={train_metrics['consistency']:.4e} p={train_metrics['physics']:.4e} | "
            f"val loss={val_metrics['loss']:.4e}"
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
