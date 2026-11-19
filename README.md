# PSGNO

Reference implementation of

> **PSGNO: A Physics-Informed Spectrally Gated Neural Operator for Noise-Robust PDE Forecasting**

PSGNO couples a U-Net denoiser with a Fourier neural operator predictor. The
predictor is equipped with a **Physics-Informed Spectral Gate (PISG)** that
attenuates Fourier modes according to the physical residual they carry, and a
**spatio-temporal Transformer** that models long-range dependencies in space
and time. Training proceeds in two stages: supervised pre-training on clean
source-domain data, followed by unsupervised fine-tuning on noisy
target-domain data with a small set of trainable parameters.

## Repository structure

```
.
├── configs/default.yaml      # all hyper-parameters
├── psgno/
│   ├── model.py              # PSGNO, FNO encoder, PISG, U-Net, Transformer
│   ├── data_utils.py         # normalization, sliding windows, domain splits
│   ├── noise.py              # structured measurement noise
│   ├── physics.py            # residual operators used by PISG
│   ├── losses.py             # denoising / prediction / consistency / physics
│   ├── utils.py              # config, seeding, checkpoints, metrics
│   ├── train_pretrain.py     # stage 1 (Algorithm 1, lines 1-12)
│   └── train_finetune.py     # stage 2 (Algorithm 1, lines 13-28)
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ and PyTorch 2.1+ are required. A CUDA-capable GPU is recommended.

## Data format

A dataset is a single NumPy array of shape `(N, T, H, W, C)`:

| axis | meaning                                              |
|------|------------------------------------------------------|
| `N`  | number of independent trajectories                    |
| `T`  | number of stored snapshots                            |
| `H,W`| spatial grid                                          |
| `C`  | state channels (`u_x, u_y` for incompressible flow)   |

Trajectories are split into a source domain (first `source_ratio` of the
trajectories, used for pre-training) and a target domain (the remainder, used
for fine-tuning).

## Usage

Stage 1 — supervised pre-training on clean source-domain data:

```bash
python -m psgno.train_pretrain --config configs/default.yaml
```

Stage 2 — unsupervised fine-tuning on noisy target-domain data:

```bash
python -m psgno.train_finetune --config configs/default.yaml
```

Both scripts write `best.pt` (weights, optimizer state and normalization
statistics) to the directory given by `save_dir` in the config file.

## Correspondence with the paper

| Paper                                       | Code                                               |
|---------------------------------------------|----------------------------------------------------|
| Noise model, Eqs. (1)-(4)                   | `noise.structured_noise`                           |
| Denoiser, Eqs. (5)-(6)                      | `model.UNetDenoiser`                               |
| Fourier layer, Eq. (7)                      | `model.SpectralConv2d`, `model.FNOEncoder`         |
| PISG gate, Eqs. (9)-(11)                    | `model.SpectralConv2d._gate`                       |
| Transformer predictor, Sec. 3.4             | `model.SpatioTemporalTransformer`                  |
| Low-rank adapters, Eq. (15)                 | `model.LoRALinear`                                 |
| Pre-training objective, Eqs. (12)-(13)      | `losses.denoising_loss`, `losses.prediction_loss`  |
| Consistency loss, Eq. (17)                  | `losses.consistency_loss`                          |
| Physics loss, Eq. (18)                      | `losses.physics_loss`                              |
| Stage-1 / stage-2 schedule, Algorithm 1     | `train_pretrain.py`, `train_finetune.py`           |

## Residual operators

All residuals of Appendices A and B are implemented in `psgno/physics.py`. Every
operator maps a spatiotemporal field `(B, T, H, W, C)` to the residual
`(B, T, H, W, R)`; the time axis is kept because each governing equation
contains `d/dt`, which also lets PISG gate every frame of the window.

| Key         | Equation                            | State variables          |
|-------------|-------------------------------------|--------------------------|
| `swe`       | Shallow water, Eq. (19)             | `h, u, v`                |
| `dr`        | FitzHugh–Nagumo, Eq. (21)           | `u, v`                   |
| `ins`       | Incompressible NS, Eq. (22)         | `u_x, u_y`               |
| `cns`       | Compressible NS, Eq. (23)           | `rho, v_x, v_y, p`       |
| `cfdbench`  | Vorticity transport, Eq. (25)       | `u_x, u_y`               |

Select one with `model.residual` and bind its coefficients through
`model.residual_params` (for example `nu`, `dt`, `gr`, `gamma`, `eta`, `zeta`).

The `ins` operator is the pressure-free form of Appendix A.3 (vorticity
transport combined with continuity). Its forcing term is read from the dataset,
because the force is re-sampled per trajectory and stored with the data: list
it in `data.aux_keys: [forcing]` and the loader passes it to the residual as
`aux["forcing"]`. The shallow-water bathymetry is handled the same way with
`bathymetry`. For the two-phase tube and dam cases set `two_phase: true` in
`residual_params`: the released arrays carry only `(u, v)`, so the interface is
inferred from the speed gradient and its band is excluded from the residual.

## Extending to a new PDE

A new PDE only requires registering an operator with the same signature:

```python
from psgno.physics import register

def my_residual(field):      # (B, T, H, W, C) -> (B, T, H, W, R)
    ...

register("my_pde", my_residual)
```

then setting `model.residual: my_pde` in the configuration file.

## Scope of this release

This repository provides the unified framework, configuration and the full
model/training skeleton. Benchmark data and the evaluation/visualization
scripts are not included.

## Citation

```bibtex
@article{mao2025psgno,
  title   = {PSGNO: A Physics-Informed Spectrally Gated Neural Operator for Noise-Robust PDE Forecasting},
  author  = {Mao, Jiaheng and Li, Ye and Cao, Yuzhe and Huang, Shengjun},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  year    = {2025}
}
```
