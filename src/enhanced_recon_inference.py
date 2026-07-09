#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator, DeepSpeedPlugin

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder2
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils
from models import *

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device
print("device:",device)


# In[2]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    model_name = "final_subj01_pretrained_40sess_24bs"
    print("model_name:", model_name)

    # global_batch_size and batch_size should already be defined in the above cells
    # other variables can be specified in the following string:
    jupyter_args = f"--model_name={model_name} --subj=1"
    print(jupyter_args)
    jupyter_args = jupyter_args.split()

    from IPython.display import clear_output # function to clear print outputs in cell
    get_ipython().run_line_magic('load_ext', 'autoreload')
    # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions
    get_ipython().run_line_magic('autoreload', '2')


# In[28]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="will load ckpt for model found in ../train_logs/model_name",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,3,4,5,6,7,8],
    help="Evaluate on which subject?",
)
parser.add_argument(
    "--seed",type=int,default=42,
)
parser.add_argument(
    "--img2img_timepoint", type=int, default=13,
    help="Number of denoising steps to retain from the end of the 25-step SDXL "
         "refinement schedule. Default 13 matches the published MindEye2 setting. "
         "Higher = more refinement / more text-prompt drift; lower = lighter "
         "refinement / closer to the input mix.",
)
if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

# seed all random functions
utils.seed_everything(seed)

# make output directory
os.makedirs("evals",exist_ok=True)
os.makedirs(f"evals/{model_name}",exist_ok=True)

# `plotting` gates which (multi-GB) tensors we bother loading below; defined here
# (was previously in cell 30) so the lazy-load block can branch on it.
plotting = False
if utils.is_interactive(): plotting=True

# Some of these files are downloadable from huggingface: https://huggingface.co/datasets/pscotti/mindeyev2/tree/main/evals
# The others are obtained from running recon_inference.ipynb first with your desired model
#
# Lazy-load policy: at 9000 training samples a pre-resized 768x768 fp32 stack
# is ~64 GB and blows the slurm cgroup. Only load tensors we actually consume,
# and resize per-sample inside the loop instead of eagerly over the whole stack.
all_recons = torch.load(f"evals/{model_name}/{model_name}_all_recons.pt") # these are the unrefined MindEye2 recons! (kept at 256 here, upsampled per-sample below)
all_predcaptions = torch.load(f"evals/{model_name}/{model_name}_all_predcaptions.pt")

# These are only consumed inside `if plotting:` blocks, so skip the (multi-GB) loads otherwise.
if plotting:
    all_images = torch.load(f"evals/all_images.pt")
    all_clipvoxels = torch.load(f"evals/{model_name}/{model_name}_all_clipvoxels.pt")
    try:
        all_blurryrecons = torch.load(f"evals/{model_name}/{model_name}_all_blurryrecons.pt")
        all_blurryrecons = transforms.Resize((768,768))(all_blurryrecons).float()
        has_blurry_recons = True
    except FileNotFoundError:
        print(f"Note: No blurry recons found (model trained with --no-blurry_recon)")
        all_blurryrecons = None
        has_blurry_recons = False
else:
    all_images = None
    all_clipvoxels = None
    all_blurryrecons = None
    has_blurry_recons = False

resize_768 = transforms.Resize((768, 768))

print(model_name)
print("all_recons", all_recons.shape, "predcaptions", all_predcaptions.shape)


# In[29]:


config = OmegaConf.load("generative_models/configs/unclip6.yaml")
config = OmegaConf.to_container(config, resolve=True)
unclip_params = config["model"]["params"]
sampler_config = unclip_params["sampler_config"]
sampler_config['params']['num_steps'] = 38
config = OmegaConf.load("generative_models/configs/inference/sd_xl_base.yaml")
config = OmegaConf.to_container(config, resolve=True)
refiner_params = config["model"]["params"]

network_config = refiner_params["network_config"]
denoiser_config = refiner_params["denoiser_config"]
first_stage_config = refiner_params["first_stage_config"]
conditioner_config = refiner_params["conditioner_config"]
scale_factor = refiner_params["scale_factor"]
disable_first_stage_autocast = refiner_params["disable_first_stage_autocast"]

# SDXL refinement base checkpoint. Defaults to zavychromaxl_v30.safetensors in
# the current directory (run from src/); override with the ZAVYCHROMAXL_PATH env var.
base_ckpt_path = os.environ.get("ZAVYCHROMAXL_PATH", "zavychromaxl_v30.safetensors")
base_engine = DiffusionEngine(network_config=network_config,
                       denoiser_config=denoiser_config,
                       first_stage_config=first_stage_config,
                       conditioner_config=conditioner_config,
                       sampler_config=sampler_config, # using the one defined by the unclip
                       scale_factor=scale_factor,
                       disable_first_stage_autocast=disable_first_stage_autocast,
                       ckpt_path=base_ckpt_path)
base_engine.eval().requires_grad_(False)
base_engine.to(device)

base_text_embedder1 = FrozenCLIPEmbedder(
    layer=conditioner_config['params']['emb_models'][0]['params']['layer'],
    layer_idx=conditioner_config['params']['emb_models'][0]['params']['layer_idx'],
)
base_text_embedder1.to(device)

base_text_embedder2 = FrozenOpenCLIPEmbedder2(
    arch=conditioner_config['params']['emb_models'][1]['params']['arch'],
    version=conditioner_config['params']['emb_models'][1]['params']['version'],
    freeze=conditioner_config['params']['emb_models'][1]['params']['freeze'],
    layer=conditioner_config['params']['emb_models'][1]['params']['layer'],
    always_return_pooled=conditioner_config['params']['emb_models'][1]['params']['always_return_pooled'],
    legacy=conditioner_config['params']['emb_models'][1]['params']['legacy'],
)
base_text_embedder2.to(device)

batch={"txt": "",
      "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
      "crop_coords_top_left": torch.zeros(1, 2).to(device),
      "target_size_as_tuple": torch.ones(1, 2).to(device) * 1024}
out = base_engine.conditioner(batch)
crossattn = out["crossattn"].to(device)
vector_suffix = out["vector"][:,-1536:].to(device)
print("crossattn", crossattn.shape)
print("vector_suffix", vector_suffix.shape)
print("---")

batch_uc={"txt": "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",
      "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
      "crop_coords_top_left": torch.zeros(1, 2).to(device),
      "target_size_as_tuple": torch.ones(1, 2).to(device) * 1024}
out = base_engine.conditioner(batch_uc)
crossattn_uc = out["crossattn"].to(device)
vector_uc = out["vector"].to(device)
print("crossattn_uc", crossattn_uc.shape)
print("vector_uc", vector_uc.shape)


# In[30]:


num_samples = 1 # PS: I tried increasing this to 16 and picking highest cosine similarity like we did in MindEye1, it didnt seem to increase eval performance!
# img2img_timepoint is set from --img2img_timepoint (default 13, matching MindEye2 published).
# Keeps the last N of 25 sigmas; higher = more refinement freedom.
# Variable is already defined via globals() from args.
base_engine.sampler.guider.scale = 5 # 5 # cfg
def denoiser(x, sigma, c): return base_engine.denoiser(base_engine.model, x, sigma, c)

if plotting or num_samples>1:
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    )
    clip_img_embedder.to(device)


# In[31]:


# Per-sample resume: scan evals/{model_name}/enhanced_parts/ for already-done
# samples, skip them, write each new sample atomically. Lets multi-pass
# chained/resumed jobs resume from prior progress.
import glob
enhanced_parts_dir = f"evals/{model_name}/enhanced_parts"
os.makedirs(enhanced_parts_dir, exist_ok=True)
existing_enhanced = {
    int(os.path.basename(p).split("_")[-1].split(".")[0])
    for p in glob.glob(f"{enhanced_parts_dir}/sample_*.pt")
}
if existing_enhanced:
    print(f"[resume] Found {len(existing_enhanced)} completed enhanced samples in "
          f"{enhanced_parts_dir}; skipping them.")

for img_idx in tqdm(range(len(all_recons))):
    if img_idx in existing_enhanced:
        continue
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), base_engine.ema_scope():
        base_engine.sampler.num_steps = 25

        # Per-sample 768 upsample (was previously done eagerly over the whole stack at load time;
        # at 9000 samples that intermediate is ~64 GB and OOMs the slurm job).
        image = resize_768(all_recons[[img_idx]]).float()

        if plotting:
            if has_blurry_recons:
                print("blur pixcorr:",utils.pixcorr(all_blurryrecons[[img_idx]].float(), all_images[[img_idx]].float()))
                print("blur cossim:",nn.functional.cosine_similarity(clip_img_embedder(utils.resize(all_blurryrecons[[img_idx]].float(),256).to(device)).flatten(1), 
                                                             clip_img_embedder(utils.resize(all_images[[img_idx]].float(),224).to(device)).flatten(1)))

            print("recon pixcorr:",utils.pixcorr(image,all_images[[img_idx]].float()))
            print("recon cossim:",nn.functional.cosine_similarity(clip_img_embedder(utils.resize(image,224).to(device)).flatten(1), 
                                                         clip_img_embedder(utils.resize(all_images[[img_idx]].float(),224).to(device)).flatten(1)))

        image = image.to(device)
        prompt = all_predcaptions[[img_idx]][0]
        # prompt = ""
        if plotting: 
            print("prompt:",prompt)
            if has_blurry_recons:
                plt.imshow(transforms.ToPILImage()(all_blurryrecons[img_idx].float()))
                plt.show()
            plt.imshow(transforms.ToPILImage()(all_recons[img_idx].float()))
            plt.show()
            plt.imshow(transforms.ToPILImage()(image[0]))
            plt.show()

        # z = torch.randn(num_samples,4,96,96).to(device)
        assert image.shape[-1]==768
        z = base_engine.encode_first_stage(image*2-1).repeat(num_samples,1,1,1)

        openai_clip_text = base_text_embedder1(prompt)
        clip_text_tokenized, clip_text_emb  = base_text_embedder2(prompt)
        clip_text_emb = torch.hstack((clip_text_emb, vector_suffix))
        clip_text_tokenized = torch.cat((openai_clip_text, clip_text_tokenized),dim=-1)
        c = {"crossattn": clip_text_tokenized.repeat(num_samples,1,1), "vector": clip_text_emb.repeat(num_samples,1)}
        uc = {"crossattn": crossattn_uc.repeat(num_samples,1,1), "vector": vector_uc.repeat(num_samples,1)}

        noise = torch.randn_like(z)
        sigmas = base_engine.sampler.discretization(base_engine.sampler.num_steps).to(device)
        init_z = (z + noise * append_dims(sigmas[-img2img_timepoint], z.ndim)) / torch.sqrt(1.0 + sigmas[0] ** 2.0)
        sigmas = sigmas[-img2img_timepoint:].repeat(num_samples,1)

        base_engine.sampler.num_steps = sigmas.shape[-1] - 1
        noised_z, _, _, _, c, uc = base_engine.sampler.prepare_sampling_loop(init_z, cond=c, uc=uc, 
                                                            num_steps=base_engine.sampler.num_steps)
        for timestep in range(base_engine.sampler.num_steps):
            noised_z = base_engine.sampler.sampler_step(sigmas[:,timestep],
                                                        sigmas[:,timestep+1],
                                                        denoiser, noised_z, cond=c, uc=uc, gamma=0)
        samples_z_base = noised_z
        samples_x = base_engine.decode_first_stage(samples_z_base)
        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

        # find best sample
        if plotting==False and num_samples==1:
            samples = samples[0]
        else:
            sample_cossim = nn.functional.cosine_similarity(clip_img_embedder(utils.resize(samples,224).to(device)).flatten(1), 
                                clip_img_embedder(utils.resize(all_images[[img_idx]].float(),224).to(device)).flatten(1))
            which_sample = torch.argmax(sample_cossim)
            best_cossim = torch.max(sample_cossim)

            if plotting:
                print("samples", samples.shape)
                for n in range(num_samples):
                    recon = transforms.ToPILImage()(samples[n])
                    plt.imshow(recon)
                    plt.show()
                    if (n==which_sample).item(): print("CHOSEN ABOVE")
                    print("upsampled pixcorr:",utils.pixcorr(samples[[n]].cpu(),all_images[[img_idx]].float()))
                    print("upsampled cossim:",nn.functional.cosine_similarity(clip_img_embedder(utils.resize(samples[[n]],224).to(device)).flatten(1), 
                                                         clip_img_embedder(utils.resize(all_images[[img_idx]].float(),224).to(device)).flatten(1)))
                err # dont want to do entire for loop with plotting=True

            samples = samples[which_sample]

        # Resize to 256 BEFORE saving so each part is ~0.79MB (vs ~7MB at 768).
        # Keeps peak assembly RAM bounded (9000 × 256² ≈ 7GB instead of ~64GB).
        samples = transforms.Resize((256, 256))(samples.cpu()[None]).float()
        # Atomic per-sample save for resume.
        part_path = f"{enhanced_parts_dir}/sample_{img_idx:05d}.pt"
        tmp_path = part_path + ".tmp"
        torch.save(samples, tmp_path)
        os.replace(tmp_path, part_path)

# Assemble all per-sample parts into the final tensor.
print(f"\n[parts] Assembling final enhanced tensor from {enhanced_parts_dir}...")
part_paths = sorted(glob.glob(f"{enhanced_parts_dir}/sample_*.pt"))
expected = len(all_recons)
if len(part_paths) != expected:
    print(f"[parts] WARNING: {len(part_paths)}/{expected} samples present. "
          f"Successor will continue. Skipping final assembly this pass.")
    sys.exit(0)
# List-then-single-vstack is O(n); avoid the prior in-loop vstack pattern.
all_enhancedrecons = torch.vstack([
    torch.load(p) for p in tqdm(part_paths, desc="loading enhanced parts")
])

print("all_enhancedrecons", all_enhancedrecons.shape)
out_path = f"evals/{model_name}/{model_name}_all_enhancedrecons.pt"
tmp_out = out_path + ".tmp"
torch.save(all_enhancedrecons, tmp_out)
os.replace(tmp_out, out_path)
print(f"saved {out_path}")

if not utils.is_interactive():
    sys.exit(0)

