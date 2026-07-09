---
license: mit
library_name: pytorch
tags:
  - fmri
  - brain-decoding
  - image-reconstruction
  - mindeye2
  - neuroscience
  - natural-scenes-dataset
---

# MindEye2 — Subject 2 checkpoints

Trained [MindEye2](https://arxiv.org/abs/2403.11207) checkpoints for **Subject 2** of the Natural Scenes Dataset (NSD). They are the base reconstruction models for the inference-time extensions in [kevin-bretz/MindEyeV2](https://github.com/kevin-bretz/MindEyeV2): a diffusion timepoint sweep, a VLM caption ablation, and Brain-Optimized Inference (BOI).

These are **inference-only** checkpoints — each file contains model weights (`model_state_dict`) only, with optimizer state removed. They contain **no NSD data**.

## Files

| Path | Model | Description |
|---|---|---|
| `finetuned_subj02_1sess_1024hid_low/last.pth` | Subject-2 fine-tune | Fine-tuned on **1 session** of Subject 2 (`hidden_dim=1024`, `n_blocks=4`, `blurry_recon=True`). This is the model the extensions run on. |
| `multisubject_excludingsubj02/last.pth` | Multi-subject pretrain | Pretrained on all NSD subjects except Subject 2; the starting point for the fine-tune above. |

## Usage

```bash
# Fine-tuned Subject-2 model (what the extensions use)
huggingface-cli download kevin-bretz/mindeye2-subj02 \
    finetuned_subj02_1sess_1024hid_low/last.pth --local-dir train_logs

# (optional) the multi-subject pretrain it was fine-tuned from
huggingface-cli download kevin-bretz/mindeye2-subj02 \
    multisubject_excludingsubj02/last.pth --local-dir train_logs
```

Then run inference (see the [repository README](https://github.com/kevin-bretz/MindEyeV2)):

```bash
python recon_inference.py --model_name=finetuned_subj02_1sess_1024hid_low --subj=2 \
    --n_blocks=4 --hidden_dim=1024 --blurry_recon --new_test
```

## Data access & license

The **weights** are released here; the underlying fMRI/stimulus **data is not** and must be obtained directly from NSD by agreeing to the [NSD Terms & Conditions](https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions). Base MindEye2 components (unCLIP decoder, converters) are on [pscotti/mindeyev2](https://huggingface.co/datasets/pscotti/mindeyev2). Please cite MindEye2, MindEye1, and the Natural Scenes Dataset.
