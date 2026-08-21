<!-- TAGNET README HEADER — Catppuccin Mocha — do not edit by hand -->
<div align="center">

[![License](https://img.shields.io/github/license/e404-tagnet/sim-AE?color=313244&labelColor=11111b&label=License&style=flat-square)](https://github.com/e404-tagnet/sim-AE/blob/main/LICENSE)
[![Status](https://img.shields.io/badge/Status-stable-a6e3a1?labelColor=11111b&style=flat-square)](https://github.com/e404-tagnet/sim-AE/pulse)
[![Version](https://img.shields.io/github/v/release/e404-tagnet/sim-AE?color=313244&labelColor=11111b&label=Version&style=flat-square)](https://github.com/e404-tagnet/sim-AE/releases)
[![Repo](https://img.shields.io/badge/Repo-sim-AE-94e2d5?labelColor=11111b&style=flat-square&logo=github&logoColor=94e2d5)](https://github.com/e404-tagnet/sim-AE)
[![Tagnet](https://img.shields.io/badge/By-Tagnet-89dceb?labelColor=11111b&style=flat-square&logo=tag&logoColor=89dceb)](https://tagnet.dev)

</div>
<!-- TAGNET README HEADER — end -->

# SimAE — Contrastive Autoencoder v1

> *Discovering self-organised latent structures without human-imposed semantics.*

SimAE replaces the KL-driven VAE objective with a contrastive (NT-Xent) loss, forcing similar sequences to cluster in a low-dimensional latent space.

## Goal

Find structure in sequential data (starting with synthetic genomic-style sequences) without labels, pre-defined categories, or human annotations.

## Quick comparison

| Aspect | HVAE v1 | SimAE v1 |
|---|---|---|
| Method | Hierarchical VAE | Contrastive Autoencoder |
| Latent | 1792 dims | 128 dims |
| Driving force | KL → N(0,1) | NT-Xent pairs |

## Key changes from HVAE

1. **Removed VAE** — KL divergence was fighting structure discovery.
2. **Added contrastive loss** — similar sequences attract in latent space.
3. **Much smaller latent** — 128 dims forces compression.
4. **Synthetic data first** — verify with known ground truth before real data.

## Architecture

```
Input (4096d) → Encoder → Latent (128d) → Decoder → Reconstruction
                              ↓
                      SimAE Loss (NT-Xent)
```

- Encoder: 4096 → 2048 → 512 → 128
- Decoder: 128 → 512 → 2048 → 4096
- Parameters: ~10.5 M

## Training strategy

1. **Phase 1:** Synthetic data with 5 known patterns (20–80 % GC).
2. **Phase 2:** Real genomic data.

## Files

- `notebooks/kaggle_complete.ipynb` — full Kaggle/Colab notebook
- `scripts/train.py` — local training script
- `configs/default.yaml` — configuration
- `TRAINING_REPORT.md` — fill-in-the-blanks report template

## Status

Early experimental phase. Phase 1 synthetic data is the current focus.

## License

This project is licensed under the [MIT License](./LICENSE).

<!-- TAGNET README FOOTER — start -->

<div align="center">

**Like this work? Fuel the next widget / experiment / scaffold.**

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-%23FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/e404.tagnet)
[![Patreon](https://img.shields.io/badge/Support-Patreon-ff424d?logo=patreon&logoColor=white&style=for-the-badge)](https://www.patreon.com/VeritasExMachina?utm_campaign=creatorshare_creator)

<small>Crafted with caffeine, curiosity, and a Catppuccin palette · © e404-tagnet</small>

</div>
<!-- TAGNET README FOOTER — end -->
