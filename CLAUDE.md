# terrain-diffusion

A from-scratch diffusion model for generating hyper-realistic terrain heightmaps, with a chunked tiling system for arbitrary-size output and exporters for Unity, Unreal, and Godot.

This repo is a learning project. The point is to understand diffusion models by implementing one end-to-end, not to match state-of-the-art quality.

## Working agreement with Claude

**Do NOT auto-edit code. Do NOT run shell commands.**

Instead:
- Tell me exactly what to write, where to write it, and why.
- For code: show the code in a fenced block with the target file path as a header. I will paste it myself.
- For commands: show the command in a fenced block and explain what it does. I will run it myself.
- For file edits: quote the existing snippet and the replacement snippet side by side.
- If multiple files need changes, list them in order with a short rationale per change.
- Ask before assuming. If a design decision is ambiguous, surface the tradeoff and let me pick.

The only tools you should use freely are Read (to look at existing files) and search tools (Grep, Glob). Avoid Edit, Write, Bash, and any tool that mutates state unless I explicitly ask.

## Hardware constraints

- **GPU**: RTX 4070 mobile, 8GB VRAM
- **Strategy**: develop and debug locally; offload long training runs to Kaggle (30h/week free, 16–32GB GPUs) and Colab as backup
- **Persistence**: checkpoints pushed to Hugging Face Hub so sessions can resume across machines
- Thermal throttling is real on laptop. Plan training to be resumable from checkpoint at any point.

## Technical decisions (already made)

- **Approach**: diffusion model trained from scratch (not GAN, not fine-tuned SD)
- **Formulation**: EDM (Karras et al. 2022) — preconditioning, log-normal σ sampling, Heun 2nd-order sampler
- **Architecture**: UNet, ~60–80M params, 4 resolution levels, attention only at 16²
- **Resolution**: progressive training 64² → 128² → 256². 256² is the final tile size.
- **Optimizer**: AdamW, lr 1e-4, weight decay 0.01, 8-bit via bitsandbytes
- **EMA**: decay 0.9999, used for all sampling
- **Mixed precision**: fp16, gradient checkpointing on
- **Conditioning**: biome class embedding + classifier-free guidance (10% drop rate, scale 2–4 at inference)
- **Tiling**: MultiDiffusion at inference for arbitrary-size output. No retraining needed for tiling.
- **Determinism**: tile-level seeding so `(world_seed, x, y)` is reproducible

## Build order (8 steps)

1. DDPM on MNIST — tiny UNet, ε-prediction. Validate the diffusion loop. (1–2 days)
2. Heightmap dataset at 64², single biome (mountains). Same DDPM code. (3 days)
3. Switch to EDM — preconditioning, log-normal σ, Heun sampler. (2 days)
4. Add EMA, fp16, gradient checkpointing. (1 day)
5. Scale to 256², full multi-biome dataset. First real run — on Kaggle. (1 week wall clock)
6. Class conditioning + classifier-free guidance for biomes. (2–3 days)
7. MultiDiffusion for arbitrary-size generation. (3 days)
8. Engine exporters (Unity / Unreal / Godot) + sample scenes. (3 days)

## Planned repo layout

```
terrain-diffusion/
  data/
    download.py        # fetch SRTM / 3DEP / Copernicus / ALOS
    tile.py            # cut into tiles, normalize
    dataset.py         # PyTorch Dataset
  model/
    unet.py            # UNet architecture
    edm.py             # preconditioning + loss
    ema.py             # EMA wrapper
  train.py             # training loop, AMP, checkpointing
  sample.py            # Heun sampler, classifier-free guidance
  tile_gen.py          # MultiDiffusion for chunked output
  export/
    unity.py
    unreal.py
    godot.py
  configs/
    mnist.yaml
    heightmap_64.yaml
    heightmap_256.yaml
  CLAUDE.md
  README.md
  pyproject.toml
```

## Data sources

- SRTM (30m global)
- USGS 3DEP (1m US lidar)
- Copernicus DEM
- OpenTopography
- ALOS World 3D

Stratified sampling across biomes: alpine, dunes, coastal, fluvial, glacial, volcanic, karst, plains, archipelagos.

Tiles stored as 16-bit single-channel `.npy` with a `manifest.json` (biome label, source, real-world scale, per-tile min/max for export rescaling).

## Engine export targets

- **Unity**: 16-bit RAW, resolution 2^n + 1
- **Unreal**: 16-bit PNG/RAW, resolution 2^n + 1
- **Godot**: EXR or PNG into HeightMapShape3D / Terrain3D addon

One Python exporter reads `.npy` + manifest and emits all three formats plus a JSON describing world scale and tile layout.

## Current status

Repo not yet created. Next concrete action: scaffold the repo and start step 1 (DDPM on MNIST).
