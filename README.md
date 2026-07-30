# Theta-NanoGPT (Vast.ai single-node)

GPT-2 (124M / 350M) training on FineWeb-10B with the [Theta](https://github.com/vaseline555/Theta)
heavy-tailed noise injector and a `{AdamW, Muon+AdamW}` optimizer grid.

This branch (`nanogpt`) is self-contained for **single-node, multi-GPU** runs on
Vast.ai (default **4× RTX 4070 Super**). All ALCF/PBS orchestration has been removed;
the `theta` package is vendored under `Theta/`.

## Quick start (Vast.ai Jupyter terminal)

```bash
git clone -b nanogpt https://github.com/vaseline555/theta.git
cd theta
bash scripts/setup_vast.sh          # deps + Theta + wandb login, creates data/ & logs/
bash download_data.sh 9             # ~2 GB smoke test  (103 = full ~21 GB)
bash run.sh configs/base_adamw.yaml 4 --wandb   # single run on 4 GPUs
```

`ngpus` defaults to 4 everywhere; pass a number to override, or `all` to use every
visible GPU. Approx wall-clock for one GPT-2 small run (5000 steps): **~1–1.5 h on 4×
RTX 4070S**.

## Disk space

Each FineWeb shard ≈ 200 MB.

| Data                | Shards | Disk    |
|---------------------|--------|---------|
| Smoke test (default)| 9 + val| ~2 GB   |
| Full FineWeb-10B    | 103+val| ~21 GB  |
| Checkpoints (small) | per run| ~1.5 GB |

Rent ~10 GB for testing, ~30 GB for a full run + checkpoints.

## Sweeps

`scripts/sweep.sh` reproduces the original grids sequentially on one node:

```bash
bash scripts/sweep.sh small_adamw 4        # lr grid, GPT-2 small AdamW
bash scripts/sweep.sh small_theta_muon 4   # config x grad_clip, Theta Muon+AdamW
# presets: small_adamw small_muon small_theta_adamw small_theta_muon
#          medium_adamw medium_theta_muon
```

Runs resume from `logs/results/<name>_latest.pt` and skip finished runs.

### Splitting a sweep across machines

Each preset is independent, so assign different presets to different boxes — e.g.
box A runs `small_adamw`, box B runs `small_muon`, box C runs `small_theta_muon`.
Every machine needs its own `setup_vast.sh` + `download_data.sh` first (data/logs
are git-ignored and local to each box). W&B ties them back together: all runs log to
project `Theta_NanoGPT` under entity `vaseline555`, grouped by preset, so the
dashboard shows them side by side regardless of which machine produced them. Run
names are deterministic (e.g. `small_adamw_lr8e-4`), so don't run the *same* preset
on two boxes or they'll collide under one W&B name.

## Layout

```
train_gpt2.py        # training entrypoint (torchrun)
configs/             # optimizer / Theta YAML presets
Theta/               # vendored theta package (pip install -e ./Theta)
download_data.sh     # FineWeb shard downloader -> ./data
run.sh               # single run
scripts/setup_vast.sh
scripts/sweep.sh
```

`data/` and `logs/` are git-ignored and created by the scripts.
