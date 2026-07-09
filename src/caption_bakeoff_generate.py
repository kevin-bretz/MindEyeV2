"""Caption-schema bake-off: Phase 1 — generate captions with Qwen2.5-VL.

Generates 7 caption variants for the same 50 NSD test-set ground-truth images.
Schema 0 reuses the existing human-written NSD COCO captions (control).
Schemas 1-6 are produced by Qwen2.5-VL-72B-Instruct-AWQ with different prompts.

Reads:
    evals/all_images.pt          — NSD test ground-truth images
    evals/all_captions.pt        — NSD COCO captions (for schema 0 control)

Writes:
    evals/caption_bakeoff/captions.json
        {
          "indices":      [0, 1, ..., 49],
          "0_nsd_gt":     ["existing COCO caption", ...],
          "1_short":      ["...", ...],
          "2_dense":      [...],
          "3_tags":       [...],
          "4_sent_tags":  [...],
          "5_style":      [...],
          "6_positional": [...],
        }

Resumable: if captions.json exists with N schemas already populated, only the
remaining schemas are regenerated.
"""
import argparse
import json
import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


SCHEMAS = {
    "1_short": (
        "Describe this image in 10 words or fewer. Subject first. "
        "Return only the description, no preamble."
    ),
    "2_dense": (
        "Describe this image in one rich sentence (max 50 words). Cover: "
        "subject, action, setting, lighting, composition, style. Avoid "
        "spatial language (left/right/foreground/background) and avoid "
        "specific counts. Subject first. "
        "Return only the description, no preamble."
    ),
    "3_tags": (
        "Describe this image as a comma-separated list of tags only. No "
        "grammar, no sentences, no preamble. Order: subject, action, "
        "attributes, setting, style, lighting. Maximum 15 tags."
    ),
    "4_sent_tags": (
        "Describe this image in one rich sentence (max 40 words) covering "
        "subject, action, setting, lighting, style. Then append a "
        "comma-separated list of 6 keywords. Format: "
        "'<sentence>, <kw1>, <kw2>, <kw3>, <kw4>, <kw5>, <kw6>'. "
        "Return only that, no preamble."
    ),
    "5_style": (
        "Describe this image in one rich sentence (max 50 words). Emphasize "
        "photographic medium, lighting quality, mood, atmosphere, color "
        "palette, and composition style. Subject is mentioned but secondary. "
        "Return only the description, no preamble."
    ),
    "6_positional": (
        "Describe this image in one sentence (max 50 words). Include "
        "explicit spatial layout: left/right, above/below, "
        "foreground/background, near/far. Make spatial relations "
        "unambiguous. Subject first, then layout, then attributes/style. "
        "Return only the description, no preamble."
    ),
}


def load_test_subset(n_images: int):
    """Load the first n_images test images + their NSD COCO captions."""
    print(f"[loading] evals/all_images.pt ...")
    all_images = torch.load("evals/all_images.pt", map_location="cpu")
    print(f"  all_images: {tuple(all_images.shape) if hasattr(all_images, 'shape') else len(all_images)}")

    print(f"[loading] evals/all_captions.pt ...")
    all_captions = torch.load("evals/all_captions.pt", map_location="cpu")
    n_total = len(all_images)
    n = min(n_images, n_total)
    indices = list(range(n))

    nsd_gt = []
    for i in indices:
        c = all_captions[i]
        if hasattr(c, "tolist"):
            c = c.tolist()
        if isinstance(c, list):
            c = c[0] if c else ""
        nsd_gt.append(str(c).strip())

    pil_images = []
    for i in indices:
        t = all_images[i]
        if t.dtype != torch.float32:
            t = t.float()
        if t.max() <= 1.5:
            t = (t * 255).clamp(0, 255).to(torch.uint8)
        else:
            t = t.clamp(0, 255).to(torch.uint8)
        pil_images.append(transforms.ToPILImage()(t))

    return indices, pil_images, nsd_gt


def generate_with_qwen(model, processor, pil_images, prompt, max_new_tokens=128):
    """Run Qwen2.5-VL on each image with `prompt` and return a list of caption strings."""
    captions = []
    device = next(model.parameters()).device
    for pil_img in tqdm(pil_images, desc="caption", leave=False):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text",  "text":  prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[pil_img], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out_ids = gen_ids[0, inputs.input_ids.shape[1]:]
        cap = processor.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        cap = cap.replace("\n", " ").strip()
        captions.append(cap)
    return captions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_id", type=str,
                    default="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
                    help="HuggingFace model id (use Qwen2.5-VL-7B-Instruct for "
                         "smaller fallback)")
    ap.add_argument("--output_dir", type=str, default="evals/caption_bakeoff")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "captions.json")

    indices, pil_images, nsd_gt = load_test_subset(args.n_images)
    n = len(indices)
    print(f"  using {n} test images: indices {indices[:5]}...{indices[-3:]}")

    captions_out = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                captions_out = json.load(f)
            print(f"[resume] loaded existing {out_path} with {len(captions_out)} keys")
        except Exception as e:
            print(f"[warn] could not load existing {out_path}: {e}; starting fresh")
            captions_out = {}

    captions_out["indices"] = indices
    captions_out["0_nsd_gt"] = nsd_gt
    print(f"  schema 0_nsd_gt: {len(nsd_gt)} captions (control, no VLM)")
    print(f"    example: {nsd_gt[0]!r}")

    todo = [k for k in SCHEMAS if k not in captions_out or len(captions_out[k]) != n]
    if not todo:
        print("All VLM schemas already complete; saving and exiting.")
        with open(out_path, "w") as f:
            json.dump(captions_out, f, indent=2)
        return

    print(f"\n[loading] {args.model_id} ...")
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _ModelCls
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as _ModelCls
        print("  note: Qwen2_5_VLForConditionalGeneration not available, "
              "using Qwen2VLForConditionalGeneration")

    # Force float16 for AWQ-quantized models. autoawq's Triton GEMM kernels
    # have fp16-only scales/zeros; loading with "auto" picks bf16 from the
    # model's config and triggers fp16/bf16 mismatch in awq_gemm_kernel.
    is_awq = "AWQ" in args.model_id.upper()
    dtype = torch.float16 if is_awq else "auto"
    model = _ModelCls.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  loaded; ~{n_params:.1f}B params on device {next(model.parameters()).device}")

    for sname in todo:
        prompt = SCHEMAS[sname]
        print(f"\n[generating] schema={sname}")
        captions = generate_with_qwen(model, processor, pil_images, prompt)
        captions_out[sname] = captions
        with open(out_path, "w") as f:
            json.dump(captions_out, f, indent=2)
        print(f"  saved (incremental). example[0]: {captions[0]!r}")

    print(f"\nWrote {out_path} ({len(captions_out)} keys)")


if __name__ == "__main__":
    main()
