# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IsoScopeXX is a deep learning framework for compression-aware isotropic super-resolution in expansion microscopy. It uses VQGAN (Vector Quantized GAN) to simultaneously enhance axial resolution (up to 8x) and compress data (~128x) without requiring high-resolution ground truth.

**Stack:** Python 3 + PyTorch + PyTorch Lightning

## Training Commands

```bash
# Basic training (all parameters from YAML config)
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj <project_name> --env <environment>

# 8X enhancement (default)
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj default/max10skip4 --env t09 --nocut

# 4X enhancement
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj default/max10skip4 --env t09 --nocut --downbranch 2 --cropz 32

# With contrastive loss
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj default/max10skip4 --env t09 --lbNCE 1

# Quick test run
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisrtest --prj test --env test --nocut
```

**Key CLI arguments:**
- `--yaml`: Config file name in `env/` (e.g., `aisr` → `env/aisr.yaml`)
- `--prj`: Project name for output directory
- `--env`: Environment from `env/env` JSON (defines LOGS and DATASET paths)
- `--models`: Override model architecture
- CLI flags override YAML defaults

## Data Preprocessing

Convert 3D TIFF volumes to training patches using `topatch.py`:

```python
from topatch import tif_to_patches
tif_to_patches([npy0],
               destination=['patches/'],
               dh=(64, 256, 256),     # patch dimensions (z, x, y)
               step=(64, 256, 256),   # sliding window step
               norm=('11',),          # normalize to [-1, 1]
               percentile=[0.1, 99.9])
```

## Architecture

**Configuration cascade:** `env/aisr.yaml` (defaults) → CLI args (overrides) → model-specific args

**Primary model:** `ae0iso0tccutvqq` - VQGAN with vector quantization
- Encoder: 2D slices → 4-channel latent codes (ldm/modules/diffusionmodules/)
- Quantizer: 256 codebook entries, 4-dim embeddings
- Decoder: Latent → reconstructed slices
- Generator (netG): 3D U-Net with skip connections (ed023e architecture)
- Discriminator (netD): PatchGAN 16x16, multi-view (6 orientations)

**Key directories:**
- `models/` - Model architectures (ae0iso0tccutvqq.py is primary)
- `networks/` - Network building blocks, loss functions, registry
- `ldm/` - VQGAN config (vqgan.yaml) and encoder/decoder modules
- `dataloader/data_multi.py` - PairedImageDataset for 3D slices
- `env/` - Config files (YAML) and environment paths (JSON)

**Loss components:** VQGAN perceptual (LPIPS) + adversarial + L1 reconstruction + VQ commitment + optional CUT contrastive

## Configuration Files

- `env/aisr.yaml` - Primary training config
- `env/aisrtest.yaml` - Quick test config
- `env/env` - JSON mapping environment names to DATASET/LOGS paths
- `ldm/vqgan.yaml` - VQGAN architecture specification

## Output Locations

Based on `--env` setting, outputs go to `{LOGS}/{dataset}/{prj}/`:
- `checkpoints/` - Model weights (every N epochs)
- `logs/` - TensorBoard logs
- `0.json` - Saved hyperparameters
