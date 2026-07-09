"""Caption-schema bake-off: Phase 2 — SDXL img2img + metric evaluation.

For each of the 7 caption schemas in evals/caption_bakeoff/captions.json,
run SDXL img2img on the same 50 brain-predicted starting images and compute
the standard MindEye2 metric suite vs ground truth.

Reads:
    evals/caption_bakeoff/captions.json
    evals/all_images.pt
    evals/{source_model}/{source_model}_all_recons.pt   (starting images)

Writes:
    evals/caption_bakeoff/refined_{schema}.pt          (refined images per schema)
    tables/caption_bakeoff.csv                         (one row per schema)
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy as sp
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim_fn
from torchvision import transforms
from torchvision.models import (alexnet, AlexNet_Weights,
                                efficientnet_b1, EfficientNet_B1_Weights)
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True

sys.path.append("generative_models/")
from omegaconf import OmegaConf
from generative_models.sgm.modules.encoders.modules import (
    FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder2,
)
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims


SCHEMA_ORDER = ["0_nsd_gt", "1_short", "2_dense", "3_tags",
                "4_sent_tags", "5_style", "6_positional"]


def build_sdxl(device):
    """Mirrors enhanced_recon_inference.py setup."""
    cfg_unclip = OmegaConf.to_container(
        OmegaConf.load("generative_models/configs/unclip6.yaml"), resolve=True)
    sampler_config = cfg_unclip["model"]["params"]["sampler_config"]
    sampler_config["params"]["num_steps"] = 38

    cfg_xl = OmegaConf.to_container(
        OmegaConf.load("generative_models/configs/inference/sd_xl_base.yaml"),
        resolve=True)
    rp = cfg_xl["model"]["params"]
    base_engine = DiffusionEngine(
        network_config=rp["network_config"],
        denoiser_config=rp["denoiser_config"],
        first_stage_config=rp["first_stage_config"],
        conditioner_config=rp["conditioner_config"],
        sampler_config=sampler_config,
        scale_factor=rp["scale_factor"],
        disable_first_stage_autocast=rp["disable_first_stage_autocast"],
        ckpt_path=os.environ.get("ZAVYCHROMAXL_PATH", "zavychromaxl_v30.safetensors"),
    )
    base_engine.eval().requires_grad_(False).to(device)

    cc = rp["conditioner_config"]
    e1 = FrozenCLIPEmbedder(
        layer=cc["params"]["emb_models"][0]["params"]["layer"],
        layer_idx=cc["params"]["emb_models"][0]["params"]["layer_idx"]).to(device)
    e2 = FrozenOpenCLIPEmbedder2(
        arch=cc["params"]["emb_models"][1]["params"]["arch"],
        version=cc["params"]["emb_models"][1]["params"]["version"],
        freeze=cc["params"]["emb_models"][1]["params"]["freeze"],
        layer=cc["params"]["emb_models"][1]["params"]["layer"],
        always_return_pooled=cc["params"]["emb_models"][1]["params"]["always_return_pooled"],
        legacy=cc["params"]["emb_models"][1]["params"]["legacy"]).to(device)

    # Conditioning suffixes (same as enhanced_recon_inference.py)
    batch = {"txt": "",
             "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
             "crop_coords_top_left": torch.zeros(1, 2).to(device),
             "target_size_as_tuple": torch.ones(1, 2).to(device) * 1024}
    out = base_engine.conditioner(batch)
    crossattn = out["crossattn"].to(device)
    vector_suffix = out["vector"][:, -1536:].to(device)

    batch_uc = dict(batch)
    batch_uc["txt"] = ("painting, extra fingers, mutated hands, poorly drawn hands, "
                       "poorly drawn face, deformed, ugly, blurry, bad anatomy, "
                       "bad proportions, extra limbs, cloned face, skinny, glitchy, "
                       "double torso, extra arms, extra hands, mangled fingers, "
                       "missing lips, ugly face, distorted face, extra legs, anime")
    out_uc = base_engine.conditioner(batch_uc)
    crossattn_uc = out_uc["crossattn"].to(device)
    vector_uc = out_uc["vector"].to(device)

    return base_engine, e1, e2, vector_suffix, crossattn_uc, vector_uc


def img2img_one(base_engine, e1, e2, vector_suffix, crossattn_uc, vector_uc,
                starting_image_768, prompt, img2img_timepoint=15,
                num_steps=25, cfg_scale=5.0, device="cuda"):
    """One forward pass through SDXL img2img. Returns [3, 768, 768] in [0,1]."""
    base_engine.sampler.guider.scale = cfg_scale
    base_engine.sampler.num_steps = num_steps

    z = base_engine.encode_first_stage(starting_image_768 * 2 - 1)

    openai_clip_text = e1(prompt)
    clip_text_tokenized, clip_text_emb = e2(prompt)
    clip_text_emb = torch.hstack((clip_text_emb, vector_suffix))
    clip_text_tokenized = torch.cat((openai_clip_text, clip_text_tokenized), dim=-1)
    c = {"crossattn": clip_text_tokenized, "vector": clip_text_emb}
    uc = {"crossattn": crossattn_uc, "vector": vector_uc}

    noise = torch.randn_like(z)
    sigmas = base_engine.sampler.discretization(base_engine.sampler.num_steps).to(device)
    init_z = (z + noise * append_dims(sigmas[-img2img_timepoint], z.ndim)) / \
             torch.sqrt(1.0 + sigmas[0] ** 2.0)
    sigmas = sigmas[-img2img_timepoint:].repeat(1, 1)

    base_engine.sampler.num_steps = sigmas.shape[-1] - 1
    noised_z, _, _, _, c, uc = base_engine.sampler.prepare_sampling_loop(
        init_z, cond=c, uc=uc, num_steps=base_engine.sampler.num_steps)
    def denoiser(x, sigma, cc): return base_engine.denoiser(base_engine.model, x, sigma, cc)
    for ts in range(base_engine.sampler.num_steps):
        noised_z = base_engine.sampler.sampler_step(
            sigmas[:, ts], sigmas[:, ts + 1],
            denoiser, noised_z, cond=c, uc=uc, gamma=0)
    samples_x = base_engine.decode_first_stage(noised_z)
    return torch.clamp((samples_x + 1.0) / 2.0, 0.0, 1.0)[0]


def two_way_id(recons, gts, model, preprocess, feature_layer, device):
    with torch.no_grad():
        preds = model(torch.stack([preprocess(r) for r in recons]).to(device))
        reals = model(torch.stack([preprocess(g) for g in gts]).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).detach().cpu().numpy()
        reals = reals.float().flatten(1).detach().cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).detach().cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).detach().cpu().numpy()
    r = np.corrcoef(reals, preds)
    r = r[:len(gts), len(gts):]
    cong = np.diag(r)
    succ = (r < cong).sum(0)
    return float(np.mean(succ) / (len(gts) - 1))


def compute_metrics(recons_256, gts_256, device):
    """recons_256, gts_256: [N, 3, 256, 256] in [0, 1]."""
    n = len(recons_256)
    out = {}

    # --- PixCorr ---
    pp = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    g_flat = pp(gts_256).reshape(n, -1).cpu().numpy()
    r_flat = pp(recons_256).reshape(n, -1).cpu().numpy()
    out["PixCorr"] = float(np.mean([np.corrcoef(g_flat[i], r_flat[i])[0, 1] for i in range(n)]))

    # --- SSIM ---
    g_gray = rgb2gray(pp(gts_256).permute(0, 2, 3, 1).cpu().numpy())
    r_gray = rgb2gray(pp(recons_256).permute(0, 2, 3, 1).cpu().numpy())
    ssim_vals = []
    for im, rec in zip(g_gray, r_gray):
        ssim_vals.append(ssim_fn(rec, im, multichannel=True, gaussian_weights=True,
                                 sigma=1.5, use_sample_covariance=False, data_range=1.0))
    out["SSIM"] = float(np.mean(ssim_vals))

    # --- AlexNet (2-way ID) ---
    alex_w = AlexNet_Weights.IMAGENET1K_V1
    alex = create_feature_extractor(alexnet(weights=alex_w),
                                    return_nodes=["features.4", "features.11"]).to(device)
    alex.eval().requires_grad_(False)
    pp_alex = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    out["AlexNet(2)"] = two_way_id(recons_256.to(device).float(), gts_256.to(device).float(),
                                   alex, pp_alex, "features.4", device)
    out["AlexNet(5)"] = two_way_id(recons_256.to(device).float(), gts_256.to(device).float(),
                                   alex, pp_alex, "features.11", device)
    del alex; torch.cuda.empty_cache()

    # --- CLIP (2-way ID) ---
    import clip
    clip_model, _ = clip.load("ViT-L/14", device=device)
    pp_clip = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])])
    out["CLIP"] = two_way_id(recons_256.to(device).float(), gts_256.to(device).float(),
                             clip_model.encode_image, pp_clip, None, device)
    del clip_model; torch.cuda.empty_cache()

    # --- EfficientNet-B (distance) ---
    eff_w = EfficientNet_B1_Weights.DEFAULT
    eff = create_feature_extractor(efficientnet_b1(weights=eff_w),
                                   return_nodes=["avgpool"])
    eff.eval().requires_grad_(False)
    pp_eff = transforms.Compose([
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    gt_f = eff(pp_eff(gts_256))["avgpool"].reshape(n, -1).cpu().numpy()
    rc_f = eff(pp_eff(recons_256))["avgpool"].reshape(n, -1).cpu().numpy()
    out["EffNet-B"] = float(np.mean([sp.spatial.distance.correlation(gt_f[i], rc_f[i])
                                     for i in range(n)]))
    del eff

    # --- SwAV (distance) ---
    swav = torch.hub.load("facebookresearch/swav:main", "resnet50")
    swav = create_feature_extractor(swav, return_nodes=["avgpool"])
    swav.eval().requires_grad_(False)
    pp_swav = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    gt_f = swav(pp_swav(gts_256))["avgpool"].reshape(n, -1).cpu().numpy()
    rc_f = swav(pp_swav(recons_256))["avgpool"].reshape(n, -1).cpu().numpy()
    out["SwAV"] = float(np.mean([sp.spatial.distance.correlation(gt_f[i], rc_f[i])
                                 for i in range(n)]))
    del swav

    # Composite score: equal-weight 5-term similarity score on (1-distance)
    # for SwAV and the similarity metrics.
    out["Composite5"] = (out["SSIM"] + out["PixCorr"] + out["AlexNet(5)"]
                        + out["CLIP"] + (1 - out["SwAV"])) / 5.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions_path", type=str,
                    default="evals/caption_bakeoff/captions.json")
    ap.add_argument("--source_model", type=str,
                    default="finetuned_subj01_40sess_1024hid_low",
                    help="Source for starting unCLIP recons (uses _all_recons.pt)")
    ap.add_argument("--n_images", type=int, default=50)
    ap.add_argument("--img2img_timepoint", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default="evals/caption_bakeoff")
    ap.add_argument("--csv_path", type=str, default="tables/caption_bakeoff.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)

    print(f"[loading] captions: {args.captions_path}")
    with open(args.captions_path) as f:
        captions = json.load(f)
    indices = captions["indices"][:args.n_images]
    n = len(indices)
    schemas = [s for s in SCHEMA_ORDER if s in captions and len(captions[s]) >= n]
    print(f"  {n} images, {len(schemas)} schemas: {schemas}")

    print(f"[loading] evals/all_images.pt")
    all_images = torch.load("evals/all_images.pt", map_location="cpu")
    gts_256 = all_images[indices].float()
    if gts_256.shape[-1] != 256:
        gts_256 = transforms.Resize(256)(gts_256)
    gts_256 = gts_256.clamp(0, 1)
    print(f"  gts_256: {tuple(gts_256.shape)}")

    src_recons_path = f"evals/{args.source_model}/{args.source_model}_all_recons.pt"
    print(f"[loading] starting recons: {src_recons_path}")
    starts = torch.load(src_recons_path, map_location="cpu").float()
    starts = starts[indices]
    if starts.shape[-1] != 768:
        starts = transforms.Resize(768)(starts)
    starts = starts.clamp(0, 1)
    print(f"  starts (768): {tuple(starts.shape)}")

    print(f"\n[loading] SDXL pipeline ...")
    base_engine, e1, e2, vector_suffix, crossattn_uc, vector_uc = build_sdxl(device)
    print(f"  loaded.")

    # --- Phase 2a: img2img per schema (resumable) ---
    refined_per_schema = {}
    for sname in schemas:
        out_path = f"{args.output_dir}/refined_{sname}.pt"
        if os.path.exists(out_path):
            r = torch.load(out_path, map_location="cpu").float()
            if r.shape[0] == n:
                refined_per_schema[sname] = r
                print(f"[resume] {sname}: loaded cached refined ({tuple(r.shape)})")
                continue
        prompts = captions[sname][:n]
        print(f"\n[img2img] schema={sname}  ({n} samples)")
        refined = torch.zeros(n, 3, 256, 256)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), base_engine.ema_scope():
            for i in tqdm(range(n), desc=sname, leave=True):
                img768 = starts[i:i+1].to(device)
                prompt = prompts[i] if prompts[i] else " "
                refined_768 = img2img_one(
                    base_engine, e1, e2, vector_suffix, crossattn_uc, vector_uc,
                    img768, prompt, img2img_timepoint=args.img2img_timepoint,
                    device=device)
                refined[i] = transforms.Resize(256)(refined_768.cpu()[None]).float()[0]
        tmp = out_path + ".tmp"
        torch.save(refined, tmp); os.replace(tmp, out_path)
        refined_per_schema[sname] = refined
        print(f"  saved {out_path}  shape={tuple(refined.shape)}")

    # Free SDXL before metric eval to avoid VRAM contention
    del base_engine, e1, e2; torch.cuda.empty_cache()

    # --- Phase 2b: metrics ---
    print(f"\n[metrics] computing for {len(refined_per_schema)} schemas vs ground truth")
    rows = []
    for sname, refined in refined_per_schema.items():
        print(f"\n  -- {sname} --")
        m = compute_metrics(refined.float(), gts_256.float(), device)
        m["schema"] = sname
        for k, v in m.items():
            if k != "schema":
                print(f"     {k:>13s}: {v:+.4f}")
        rows.append(m)

    df = pd.DataFrame(rows)
    cols = (["schema", "Composite5"] + [c for c in df.columns
                                        if c not in ("schema", "Composite5")])
    df = df[cols]
    df = df.sort_values("Composite5", ascending=False)
    print(f"\n=== Caption bake-off results (sorted by Composite5) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    df.to_csv(args.csv_path, index=False)
    print(f"\nWrote {args.csv_path}")


if __name__ == "__main__":
    main()
