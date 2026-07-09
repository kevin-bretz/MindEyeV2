#!/usr/bin/env python
# coding: utf-8
"""
Brain-optimized inference for MindEye2 — Latent Space Scoring variant.

Same as brain_optimized_inference.py but instead of comparing GNet predictions
to measured fMRI in raw voxel space (Pearson correlation), we project both
through MindEye2's ridge regression into the shared latent space and compare
with cosine similarity there.

Outputs:
  evals/{model_name}/{model_name}_all_brain_opt_recons_v2.pt
"""

import os
import sys
import argparse
import numpy as np
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from accelerate import Accelerator
import webdataset as wds

sys.path.append('generative_models/')
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.modules.encoders.modules import (
    FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder2
)
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf

torch.backends.cuda.matmul.allow_tf32 = True

import utils
from models import GNet8_Encoder

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device
print("device:", device)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Brain-optimized inference — latent space scoring")
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--data_path", type=str, default=os.getcwd())
parser.add_argument("--cache_dir", type=str, default=os.getcwd())
parser.add_argument("--subj", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument("--new_test", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_candidates", type=int, default=8)
parser.add_argument("--n_iterations", type=int, default=5)
parser.add_argument("--convergence_tol", type=float, default=0.05)

if utils.is_interactive():
    model_name = "finetune_subj02_150ep"
    jupyter_args = f"--model_name={model_name} --subj=2"
    args = parser.parse_args(jupyter_args.split())
else:
    args = parser.parse_args()

for attr in vars(args):
    globals()[attr] = getattr(args, attr)

utils.seed_everything(seed)
os.makedirs("evals", exist_ok=True)
os.makedirs(f"evals/{model_name}", exist_ok=True)


# ---------------------------------------------------------------------------
# Load saved MindEye2 outputs
# ---------------------------------------------------------------------------
all_recons = torch.load(f"evals/{model_name}/{model_name}_all_enhancedrecons.pt")
all_predcaptions = torch.load(f"evals/{model_name}/{model_name}_all_predcaptions.pt")
all_recons = transforms.Resize((768, 768))(all_recons).float()
print(f"Loaded {len(all_recons)} initial reconstructions")


# ---------------------------------------------------------------------------
# Load test voxels and build per-image averaged betas
# ---------------------------------------------------------------------------
with h5py.File(f'{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5', 'r') as f:
    betas = torch.Tensor(f['betas'][:]).cpu()
num_voxels = betas.shape[-1]

if not new_test:
    num_test = {3: 2113, 4: 1985, 6: 2113, 8: 1985}.get(subj, 2770)
    test_url = f"{data_path}/wds/subj0{subj}/test/0.tar"
else:
    num_test = {3: 2371, 4: 2188, 6: 2371, 8: 2188}.get(subj, 3000)
    test_url = f"{data_path}/wds/subj0{subj}/new_test/0.tar"

def my_split_by_node(urls): return urls
test_data = (
    wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node)
    .decode("torch")
    .rename(behav="behav.npy", past_behav="past_behav.npy",
            future_behav="future_behav.npy", olds_behav="olds_behav.npy")
    .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
)
test_dl = torch.utils.data.DataLoader(
    test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True
)

test_images_idx = []
test_voxels = None
for behav, *_ in test_dl:
    vox = betas[behav[:, 0, 5].cpu().long()]
    test_images_idx = np.append(test_images_idx, behav[:, 0, 0].cpu().numpy())
    test_voxels = vox if test_voxels is None else torch.vstack((test_voxels, vox))
test_images_idx = test_images_idx.astype(int)

uniq_imgs = np.unique(test_images_idx)
test_voxels_averaged = torch.zeros(len(uniq_imgs), num_voxels)
for i, uniq_img in enumerate(uniq_imgs):
    locs = np.where(test_images_idx == uniq_img)[0]
    if len(locs) == 1:
        locs = locs.repeat(3)
    elif len(locs) == 2:
        locs = np.concatenate([locs, locs[:1]])
    test_voxels_averaged[i] = test_voxels[locs[:3]].mean(0)

print(f"Unique test images: {len(uniq_imgs)}, voxels per image: {num_voxels}")

# Pre-compute measured std over full voxels for convergence criterion
measured_stds = test_voxels_averaged.std(dim=1)          # (num_images,)


# ---------------------------------------------------------------------------
# Load GNet brain encoder
# ---------------------------------------------------------------------------
GNet = GNet8_Encoder(
    subject=subj,
    device=str(device),
    model_path=f"{cache_dir}/gnet_multisubject.pt"
)
print("GNet encoder loaded")


# ---------------------------------------------------------------------------
# Load MindEye2 ridge regression weights from checkpoint
# ---------------------------------------------------------------------------
print("Loading MindEye2 ridge regression weights...")
ckpt_path = os.path.join(cache_dir, '..', 'train_logs', model_name, 'last.pth')
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
state = ckpt['model_state_dict']

ridge_weight = state['ridge.linears.0.weight'].to(device).float()  # (hidden_dim, num_voxels)
ridge_bias   = state['ridge.linears.0.bias'].to(device).float()    # (hidden_dim,)
print(f"Ridge weight shape: {ridge_weight.shape}")

del ckpt, state
torch.cuda.empty_cache()

def apply_ridge(voxels):
    """
    Project voxels into MindEye2 shared latent space.
    voxels: (num_voxels,) or (N, num_voxels) float tensor on device
    returns: (hidden_dim,) or (N, hidden_dim)
    """
    return F.linear(voxels.float(), ridge_weight, ridge_bias)


# ---------------------------------------------------------------------------
# Load SDXL diffusion engine
# ---------------------------------------------------------------------------
unclip_cfg = OmegaConf.to_container(
    OmegaConf.load("generative_models/configs/unclip6.yaml"), resolve=True
)
sampler_config = unclip_cfg["model"]["params"]["sampler_config"]
sampler_config["params"]["num_steps"] = 38

xl_cfg = OmegaConf.to_container(
    OmegaConf.load("generative_models/configs/inference/sd_xl_base.yaml"), resolve=True
)
xl_params = xl_cfg["model"]["params"]
conditioner_config = xl_params["conditioner_config"]

base_ckpt = f"{cache_dir}/zavychromaxl_v30.safetensors"
if not os.path.isfile(base_ckpt):
    raise FileNotFoundError(f"SDXL checkpoint not found at {base_ckpt}.")

base_engine = DiffusionEngine(
    network_config=xl_params["network_config"],
    denoiser_config=xl_params["denoiser_config"],
    first_stage_config=xl_params["first_stage_config"],
    conditioner_config=conditioner_config,
    sampler_config=sampler_config,
    scale_factor=xl_params["scale_factor"],
    disable_first_stage_autocast=xl_params["disable_first_stage_autocast"],
    ckpt_path=base_ckpt,
)
base_engine.eval().requires_grad_(False).to(device)

text_emb1 = FrozenCLIPEmbedder(
    layer=conditioner_config["params"]["emb_models"][0]["params"]["layer"],
    layer_idx=conditioner_config["params"]["emb_models"][0]["params"]["layer_idx"],
).to(device)

text_emb2 = FrozenOpenCLIPEmbedder2(
    arch=conditioner_config["params"]["emb_models"][1]["params"]["arch"],
    version=conditioner_config["params"]["emb_models"][1]["params"]["version"],
    freeze=conditioner_config["params"]["emb_models"][1]["params"]["freeze"],
    layer=conditioner_config["params"]["emb_models"][1]["params"]["layer"],
    always_return_pooled=conditioner_config["params"]["emb_models"][1]["params"]["always_return_pooled"],
    legacy=conditioner_config["params"]["emb_models"][1]["params"]["legacy"],
).to(device)

_size_meta = lambda: {
    "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
    "crop_coords_top_left": torch.zeros(1, 2).to(device),
    "target_size_as_tuple": torch.ones(1, 2).to(device) * 1024,
}
out_uc = base_engine.conditioner({
    "txt": "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, "
           "deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, "
           "skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, "
           "missing lips, ugly face, distorted face, extra legs, anime",
    **_size_meta()
})
crossattn_uc = out_uc["crossattn"].to(device)
vector_uc = out_uc["vector"].to(device)
vector_suffix = base_engine.conditioner({"txt": "", **_size_meta()})["vector"][:, -1536:].to(device)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _denoiser(x, sigma, c):
    return base_engine.denoiser(base_engine.model, x, sigma, c)


def img2img_sample(image_768, prompt, n, img2img_timepoint, cfg_scale=5.0):
    assert image_768.shape[-1] == 768
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), base_engine.ema_scope():
        base_engine.sampler.num_steps = 25
        base_engine.sampler.guider.scale = cfg_scale
        z = base_engine.encode_first_stage(image_768 * 2 - 1).repeat(n, 1, 1, 1)
        openai_emb = text_emb1(prompt)
        clip_tok, clip_emb = text_emb2(prompt)
        clip_emb = torch.hstack((clip_emb, vector_suffix))
        clip_tok = torch.cat((openai_emb, clip_tok), dim=-1)
        c  = {"crossattn": clip_tok.repeat(n, 1, 1), "vector": clip_emb.repeat(n, 1)}
        uc = {"crossattn": crossattn_uc.repeat(n, 1, 1), "vector": vector_uc.repeat(n, 1)}
        noise = torch.randn_like(z)
        sigmas = base_engine.sampler.discretization(base_engine.sampler.num_steps).to(device)
        init_z = (z + noise * append_dims(sigmas[-img2img_timepoint], z.ndim)) / \
                 torch.sqrt(1.0 + sigmas[0] ** 2.0)
        sigmas_run = sigmas[-img2img_timepoint:].repeat(n, 1)
        base_engine.sampler.num_steps = sigmas_run.shape[-1] - 1
        noised_z, _, _, _, c, uc = base_engine.sampler.prepare_sampling_loop(
            init_z, cond=c, uc=uc, num_steps=base_engine.sampler.num_steps
        )
        for t in range(base_engine.sampler.num_steps):
            noised_z = base_engine.sampler.sampler_step(
                sigmas_run[:, t], sigmas_run[:, t + 1],
                _denoiser, noised_z, cond=c, uc=uc, gamma=0
            )
        samples_x = base_engine.decode_first_stage(noised_z)
        return torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def score_candidates(candidates, measured_voxels):
    """
    Score candidates in MindEye2 latent space.

    candidates      : (N, 3, H, W) float [0,1] CPU tensor
    measured_voxels : (num_voxels,) full voxel vector for this image

    Steps:
      1. GNet predicts voxels for each candidate image
      2. Both predicted and real voxels are projected through MindEye2 ridge
         into the shared latent space
      3. Cosine similarity in latent space is the score

    Returns:
        scores     : (N,) cosine similarity — higher = better
        pred_stds  : (N,) std of raw GNet predictions (for convergence check)
        beta_primes: (N, num_voxels) raw GNet predictions
    """
    pil_images = [
        transforms.ToPILImage()(candidates[i].clamp(0, 1))
        for i in range(len(candidates))
    ]
    beta_primes = GNet.predict(pil_images).float()           # (N, num_voxels)

    with torch.no_grad():
        # Project real voxels into latent space — once per image
        z_real = apply_ridge(measured_voxels.to(device))    # (hidden_dim,)
        z_real = z_real / z_real.norm().clamp(min=1e-8)

        # Project all candidate predicted voxels in one batch
        z_pred = apply_ridge(beta_primes.to(device))        # (N, hidden_dim)
        z_pred = z_pred / z_pred.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # Cosine similarity
        scores = (z_pred * z_real.unsqueeze(0)).sum(dim=1).cpu()  # (N,)

    pred_stds = beta_primes.std(dim=1)                       # (N,) for convergence
    return scores, pred_stds, beta_primes


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------
TIMEPOINT_SCHEDULE = [12, 9, 6, 4, 3][:n_iterations]


# ---------------------------------------------------------------------------
# Main loop with checkpointing and auto-resume
# ---------------------------------------------------------------------------
resume_path = f"evals/{model_name}/{model_name}_all_brain_opt_recons_v2.pt"

if os.path.exists(resume_path):
    all_brain_opt_recons = torch.load(resume_path)
    start_idx = len(all_brain_opt_recons)
    print(f"Resuming from image {start_idx}/{len(uniq_imgs)}")
else:
    all_brain_opt_recons = None
    start_idx = 0

for img_idx in tqdm(range(start_idx, len(uniq_imgs)), desc="BOI latent"):

    measured_voxels = test_voxels_averaged[img_idx]     # (num_voxels,) full voxels
    measured_std    = measured_stds[img_idx].item()

    current_image = all_recons[[img_idx]]               # (1, 3, 768, 768)
    prompt        = all_predcaptions[[img_idx]][0]
    best_image    = current_image.clone()

    for timepoint in TIMEPOINT_SCHEDULE:
        candidates = img2img_sample(
            current_image.to(device), prompt, n_candidates, timepoint
        )

        scores, pred_stds, _ = score_candidates(candidates, measured_voxels)

        best_idx   = scores.argmax().item()
        best_image = candidates[[best_idx]]

        if measured_std > 0:
            rel_diff = abs(measured_std - pred_stds[best_idx].item()) / measured_std
            if rel_diff < convergence_tol:
                break

        current_image = best_image

    all_brain_opt_recons = (
        best_image if all_brain_opt_recons is None
        else torch.vstack((all_brain_opt_recons, best_image))
    )

    # Save checkpoint every 50 images
    if (img_idx + 1) % 50 == 0:
        torch.save(all_brain_opt_recons, resume_path)
        print(f"Checkpoint saved at {img_idx + 1}/{len(uniq_imgs)}")


# ---------------------------------------------------------------------------
# Save final output
# ---------------------------------------------------------------------------
all_brain_opt_recons = transforms.Resize((256, 256))(all_brain_opt_recons).float()
print("Shape:", all_brain_opt_recons.shape)

torch.save(all_brain_opt_recons, resume_path)
print(f"Saved {resume_path}")

if not utils.is_interactive():
    sys.exit(0)