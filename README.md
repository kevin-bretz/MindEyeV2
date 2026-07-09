# MindEye2

**Paper (ICML 2024): https://arxiv.org/abs/2403.11207**

![](figs/recon_comparison_small_alt.png)<br>

---

## Extensions in this fork

This fork extends MindEye2 with four additions, studied on **Subject 2** of the Natural Scenes Dataset. The base MindEye2 pipeline is unchanged; each extension is additive and documented with reproduction commands under [Reproducing the extensions](#reproducing-the-extensions).

| Extension | Stage | Idea |
|---|---|---|
| **Semantic Auxiliary Loss (SemAux)** | fine-tuning | A CLIP text-space regularizer added to the fine-tuning loss, pulling a projector head on the backbone output toward a frozen CLIP-text embedding of a weak semantic label of the viewed image. |
| **Brain-Optimized Inference (BOI / L-BOI / L-BOI v2)** | inference | Generate several SDXL image-to-image candidates and keep the one whose GNet-predicted fMRI response best matches the measured response — scored in raw voxel space (BOI) or in MindEye2's ridge-regression latent space (L-BOI). |
| **Diffusion timepoint sweep** | inference | Initialize the unCLIP and SDXL refinement stages from the blurry reconstruction (encoded through the SDXL VAE) instead of pure noise, sweeping the start timepoints `T_unCLIP` and `T_SDXL`. |
| **VLM caption ablation** | inference | Replace MindEye2's GIT refinement caption with Qwen2.5-VL captions under six prompting schemas, all captioning the ground-truth image as a best-case test of the caption source. |

---

> **Running on an HPC cluster?** This `main` branch is cluster-agnostic. Ready-to-run SLURM batch scripts, an ALICE module-loading `setup.sh`, and an ALICE quick-start live on the [`alice`](../../tree/alice) branch.

## Quick start

```bash
# 1. Clone
git clone https://github.com/kevin-bretz/MindEyeV2.git
cd MindEyeV2/src

# 2. Create the fmri venv + install pinned dependencies. Needs Python 3.11 and
#    CUDA 12.1 on PATH — load them via your environment modules first if on HPC.
source setup.sh

# 3. Download the NSD subset + frozen checkpoints (run from inside src/).
python download_data.py
```

`setup.sh` creates the `fmri` virtualenv inside `src/`, installs every pinned package from `requirements.txt`, installs `dalle2-pytorch` with `--no-deps` (to avoid a PyTorch version conflict), and runs a `torch` / `sgm` import smoke test.

## External models and data

`download_data.py` pulls from [`pscotti/mindeyev2`](https://huggingface.co/datasets/pscotti/mindeyev2) into whatever directory you run it from (so run it in `src/`). Everything lands flat in `src/`, which is also the default for both `--data_path` and `--cache_dir` in the training/inference scripts.

### Trained model checkpoints

The Subject-2 fine-tuned MindEye2 checkpoint used by the extensions — and the multi-subject pretrain it was fine-tuned from — are on the Hugging Face Hub as inference-only weights (no NSD data):

```bash
# run from the repo root (MindEyeV2/)
huggingface-cli download kevin-bretz/mindeye2-subj02 \
    finetuned_subj02_1sess_1024hid_low/last.pth --local-dir train_logs
```

This lands the checkpoint at `train_logs/finetuned_subj02_1sess_1024hid_low/last.pth`, ready for `recon_inference.py --model_name=finetuned_subj02_1sess_1024hid_low`.

**Downloaded automatically into `src/`:**

| File | Role |
|------|------|
| `sd_image_var_autoenc.pth` | Diffusers VAE used by the low-level submodule |
| `convnext_xlarge_alpha0.75_fullckpt.pth` | ConvNeXt features for the blurry reconstruction loss |
| `bigG_to_L_epoch8.pth` | OpenCLIP-bigG → ViT-L linear converter for unCLIP |
| `unclip6_epoch0_step110000.ckpt` | SDXL unCLIP checkpoint (the main reconstruction decoder) |
| `wds/subj0{1..8}/` | Webdataset `.tar` files (~15 GB/subject) |
| `betas_all_subj0#_fp32_renorm.hdf5` | Preprocessed fMRI voxel betas |
| `coco_images_224_float16.hdf5` | 224×224 COCO stimulus images |
| `evals/all_images.pt`, `evals/all_captions.pt`, `evals/all_git_generated_captions.pt` | Ground-truth tensors for evaluation |

**NOT downloaded — fetch only if you need them:**

- **Pretrained MindEye2 checkpoints** (for finetuning or inference without running pretraining). `download_data.py` excludes `train_logs/` entirely. Grab whichever model you need from [huggingface.co/datasets/pscotti/mindeyev2/tree/main/train_logs](https://huggingface.co/datasets/pscotti/mindeyev2/tree/main/train_logs) and place the folder under `../train_logs/` (i.e. `MindEyeV2/train_logs/`). Example:
  ```bash
  # from MindEyeV2/
  mkdir -p train_logs && cd train_logs
  huggingface-cli download pscotti/mindeyev2 --repo-type dataset \
      --include "train_logs/multisubject_subj01_1024hid_nolow_300ep/*" \
      --local-dir .
  mv train_logs/multisubject_subj01_1024hid_nolow_300ep . && rmdir train_logs
  ```
  Then pass `--multisubject_ckpt=../train_logs/multisubject_subj01_1024hid_nolow_300ep` to `Train.py`.

- **`zavychromaxl_v30.safetensors`** (only required by `enhanced_recon_inference.py`, the optional SDXL refinement stage). Not on the HF dataset — download it from [Civitai](https://civitai.com/models/119229) (v3.0) and drop it into `src/` next to the other `.pth`/`.ckpt` files. The script now reads it from `--cache_dir` (defaults to the current directory), so no code edits are needed.

**Fetched automatically at runtime by `transformers`** (needs internet on the machine, or warm the cache beforehand): `microsoft/git-large-coco`, `openai/clip-vit-base-patch32`, `openai/clip-vit-large-patch14`.

## Installation

1. Agree to the Natural Scenes Dataset's [Terms and Conditions](https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions) and fill out the [NSD Data Access form](https://forms.gle/xue2bCdM9LaFNMeb7)

2. Git clone this repository:

```
git clone https://github.com/kevin-bretz/MindEyeV2.git
cd MindEyeV2/src
```

3. Download https://huggingface.co/datasets/pscotti/mindeyev2 contents into the src folder from your git clone.

Warning: Cloning the entire huggingface dataset will be over 100 GB of data!

The below code will download the subset of files required to run all our training / inference / evaluation code (does not download pretrained models).

```
import os
from huggingface_hub import list_repo_files, hf_hub_download

repo_id, branch, exclude_dirs, exclude_files = "pscotti/mindeyev2", "main", ["train_logs", "evals"], ["human_trials_mindeye2.ipynb", "subj01_annots.npy", "shared1000.npy"]

include_specific_files = ["evals/all_images.pt", "evals/all_captions.pt", "evals/all_git_generated_captions.pt"]

def download_files(repo_id, branch, exclude_dirs, exclude_files, include_specific_files):
    files = list_repo_files(repo_id, repo_type="dataset", revision=branch)
    for file_path in files:
        if (not any(ex_dir in file_path for ex_dir in exclude_dirs) or file_path in include_specific_files) and not any(ex_file in file_path for ex_file in exclude_files):
            hf_hub_download(repo_id, filename=file_path, repo_type="dataset", revision=branch, local_dir=os.getcwd())

download_files(repo_id, branch, exclude_dirs, exclude_files, include_specific_files)
```

4. Run ```. setup.sh``` to install a new "fmri" virtual environment. Make sure the virtual environment is activated with "source fmri/bin/activate".

## Usage

MindEye2 consists of four main jupyter notebooks: `Train.ipynb`, `recon_inference.ipynb`, `enhanced_recon_inference.ipynb`, and `final_evaluations.ipynb`. 

These files can be run as Jupyter notebooks or can be converted to .py files with configuration specified via argparser. 

If you are training MindEye2 on a single GPU on the full 40 sessions, expect that pre-training and fine-tuning both take approximately 1 day to complete.

- ```src/Train.ipynb``` trains/fine-tunes models (single-subject or multi-subject depending on your config). Check the argparser arguments to specify how you want to train the model (e.g., ```--num_sessions=1``` to train with 1-hour of data).
    - Final models used in the paper were trained on an 8xA100 80GB node and will OOM on weaker compute. You can train the model on weaker compute with very minimal performance impact by changing certain model arguments: We recommend lowering hidden_dim to 1024 (or even 512), removing the low-level submodule (``--no-blurry_recon``), and lowering the batch size.
    - To train a single-subject model, set ```--no-multi_subject``` and ```--subj=#``` where # is the subject from NSD you wish to train
    - To train a multi-subject model (i.e., pretraining), set ```--multi_subject``` and ```--subj=#``` where # is the one subject out of 8 NSD subjects to **not** include in the pretraining.
    - To fine-tune from a multi-subject model, set ```--no-multi_subject``` and ```--multisubject_ckpt=path_to_your_pretrained_ckpt_folder```
    - Note if you are running multi-gpu, you need to first set your accelerate to use deepspeed stage 2 (with cpu offloading) via "accelerate config" in terminal ([example](https://i.imgur.com/iIbvcPq.png))
- ```src/recon_inference.ipynb``` will run inference on a pretrained model, outputting tensors of reconstructions/predicted captions/etc.
- ```src/enhanced_recon_inference.ipynb``` will run the refinement stage for producing better looking reconstructions. These refined reconstructions are saved as *enhancedrecons.pt in the same folder used by recon_inference.ipynb. The unrefined reconstructions were saved as *recons.pt as part of the recon_inference.ipynb notebook.
- ```src/final_evaluations.ipynb``` will visualize the saved reconstructions and compute quantitative metrics.
- Each stage can be run directly as a `.py` script with `argparser` flags (see below). Ready-to-run SLURM batch scripts for the ALICE cluster are on the [`alice`](../../tree/alice) branch.

## Reproducing the extensions

All four extensions were evaluated on **Subject 2**, using the single-session fine-tuned baseline `finetuned_subj02_1sess_1024hid_low` as the starting reconstruction. The commands below are the portable `python` entry points; equivalent one-command SLURM batch scripts for the ALICE cluster are on the [`alice`](../../tree/alice) branch. Run them from `src/` with the `fmri` venv activated.

### 1. Semantic Auxiliary Loss (SemAux)

SemAux adds a semantic regularizer to the subject-specific fine-tuning objective. A small projector head pools the MindEye2 backbone output and maps it into CLIP ViT-L/14 text-embedding space; the target is the frozen CLIP-text embedding of a weak semantic label of the viewed image. The auxiliary term is the cosine distance to that target, added to the MindEye2 loss with weight `λ_sem = 0.05`:

```
L_total = L_MindEye2 + 0.05 · L_SemAux
```

Setup used in the report: Subject 2, 1 training session, 150 epochs, global batch size 24, learning rate 3e-4.

> **Note:** SemAux is a fine-tuning-time change that is **not part of this code snapshot** (it was developed separately from this repository). Its method and settings are documented here for completeness; the reproduction commands below cover the three inference-time extensions, whose code is included.

### 2. Brain-Optimized Inference (BOI, L-BOI, L-BOI v2)

Inference-time candidate selection over a trained MindEye2 model — the weights are not modified. From an initial reconstruction, each iteration generates `N` SDXL image-to-image candidates at a fixed noise level, scores each with a GNet forward encoder against the measured fMRI response, and keeps the best candidate as the next iteration's starting point.

- **BOI Voxel** — scores in raw voxel space: `brain_optimized_inference.py`
- **L-BOI / L-BOI v2** — scores in MindEye2's 1024-d ridge-regression latent space: `brain_optimized_inference_latent.py` (v2 starts from the *refined* recons with a gentler, shorter noise schedule)

```bash
# L-BOI v2 (Subject 2): N=8 candidates, T=3 iterations
python brain_optimized_inference_latent.py \
    --model_name=finetuned_subj02_1sess_1024hid_low --subj=2 --new_test \
    --n_candidates=8 --n_iterations=3 --convergence_tol=0.05
```

Per-variant candidate count `N`, iteration count `T`, and noise schedule:

| Variant | N | T | Noise schedule | Initialization |
|---|---|---|---|---|
| BOI Voxel | 8 | 5 | [20, 15, 12, 9, 6] | unrefined recons |
| L-BOI | 8 | 5 | [20, 15, 12, 9, 6] | unrefined recons |
| L-BOI v2 | 8 | 3 | [12, 9, 6] | refined recons |

Requires the GNet encoder weights `gnet_multisubject.pt` in `--cache_dir`.

### 3. Diffusion timepoint sweep

Instead of starting the unCLIP stage from pure Gaussian noise, initialize it from the blurry reconstruction (encoded through the SDXL VAE) and begin denoising at an intermediate timepoint `T_unCLIP` (`recon_inference.py --use_img2img_init --img2img_timepoint`). The SDXL refinement stage exposes the same control as `T_SDXL` (`enhanced_recon_inference.py --img2img_timepoint`, default 13). Lower values preserve more of the brain-derived structure; higher values give the diffusion prior more freedom.

```bash
# T_unCLIP sweep (unCLIP init timepoint), with T_SDXL at its default of 13:
for tp in 10 15 20 25 30 35; do
  python recon_inference.py --model_name=subj02_tp${tp} --subj=2 \
      --n_blocks=4 --hidden_dim=1024 --blurry_recon --new_test \
      --use_img2img_init --img2img_timepoint=${tp}
  python enhanced_recon_inference.py --model_name=subj02_tp${tp} --subj=2
  python final_evaluations.py        --model_name=subj02_tp${tp} --subj=2
done

# T_SDXL sweep (SDXL refinement timepoint) at T_unCLIP=15, reusing its unCLIP recons:
for sdxl in 8 13 18 23; do
  python enhanced_recon_inference.py --model_name=subj02_tp15 --subj=2 --img2img_timepoint=${sdxl}
  python final_evaluations.py        --model_name=subj02_tp15 --subj=2
done
```

The reported noise-init baseline is a standard run without `--use_img2img_init`. Each `subj02_tp*` `model_name` should alias your trained Subject-2 checkpoint under `train_logs/` (the ALICE scripts do this with a symlink).

### 4. VLM caption ablation

Swap the caption used during SDXL refinement. Both captioners see the ground-truth image, so only the caption text varies across rows. Schema 0 is the NSD ground-truth caption; schemas 1–6 use Qwen2.5-VL-72B-Instruct-AWQ (short, dense, tags, sentence+tags, style, positional); the GIT row uses MindEye2's frozen GIT model on the ground-truth image embedding. All rows refine at `T_SDXL = 15`.

```bash
# 1. Generate captions (7 schemas × 50 images). Needs a Qwen2.5-VL environment
#    (transformers, qwen-vl-utils, autoawq); the alice branch provisions one.
python caption_bakeoff_generate.py --n_images=50 --model_id=Qwen/Qwen2.5-VL-72B-Instruct-AWQ

# 2. Re-refine the Subject-2 recons per caption at T_SDXL=15 and score
python caption_bakeoff_render.py \
    --captions_path=evals/caption_bakeoff/captions.json \
    --source_model=finetuned_subj02_1sess_1024hid_low \
    --n_images=50 --output_dir=evals/caption_bakeoff_subj02 \
    --csv_path=tables/caption_bakeoff_subj02.csv
```

The leaderboard is written to `tables/caption_bakeoff_subj02.csv`. Requires the SDXL refinement checkpoint `zavychromaxl_v30.safetensors` (see [External models](#external-models-and-data)).

## FAQ

### What are the main differences between this and MindEye1?

MindEye2 achieves SOTA reconstruction and retrieval performance compared to past work (including MindEye1). MindEye2 also excels in low-sample settings, with good performance even with just 1 hour of training data by first pretraining the model on other participants data. MindEye2 also releases a SOTA unCLIP model (unclip6_epoch0_step110000.ckpt) by fine-tuning SDXL; this raises the ceiling performance possible for reconstructing images from CLIP image latents. MindEye2 training is also more flexible thanks to our updated webdataset approach that allows one to easily obtain the brain activations corresponding to the current sample's previous/future timepoints, brain activations from other timepoints looking at the same image, and behavioral information (button press, reaction time, etc.). 

### Where are the pretrained models? What are their configs?

The pretrained models can be downloaded from huggingface (https://huggingface.co/datasets/pscotti/mindeyev2/tree/main/train_logs) and contain various model checkpoints following pre-training and following fine-tuning.

`final_multisubject_subj0#` refer to ckpts after pre-training MindEye2 on all subjects except for the subject listed in the filename. E.g., `final_multisubject_subj01` is the model pre-trained on subjects 2, 3, 4, 5, 6, 7, and 8 from NSD. Below are some additional details for the configs used in argparser when training the model:

```
accelerate launch --mixed_precision=fp16 Train.py --model_name=final_multisubject_subj0# --multi_subject --subj=# --batch_size=42 --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --blurry_recon --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=4096 --num_sessions=40
```

`final_subj0#_pretrained_40sess_24bs` refer to ckpts after fine-tuning MindEye2 on the training data for the subject listed in the filename, initializing the starting point of the model from the ckpt saved from `final_multisubject_subj0#`.

```
accelerate launch --mixed_precision=fp16 Train.py --model_name=final_subj0#_pretrained_40sess_24bs --no-multi_subject --subj=# --batch_size=24 --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --blurry_recon --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=4096 --num_sessions=40 --multisubject_ckpt=../train_logs/final_multisubject_subj0#
```

`final_subj0#_pretrained_1sess_24bs` refer to the same procedure as above but fine-tuned on only the first session of the subject's data. 

```
accelerate launch --mixed_precision=fp16 Train.py --model_name=final_subj0#_pretrained_1sess_24bs --no-multi_subject --subj=# --batch_size=24 --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --blurry_recon --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=4096 --num_sessions=1 --multisubject_ckpt=../train_logs/final_multisubject_subj0#
```

`multisubject_subj01_1024hid_nolow_300ep` is the same as `final_multisubject_subj01` but pretrained using a less intensive pipeline where the low-level module was disabled and the hidden dimensionality was lowered from 4096 to 1024. These changes very minimally affected reconstruction and retrieval performance metrics and have the benefit of being much less computationally intensive to train. We set num_epochs=300 for this model but I do not think it would have made any difference if we had set it to num_epochs=150 instead, like the above models.

```
accelerate launch --mixed_precision=fp16 Train.py --model_name=multisubject_subj01_1024hid_nolow_300ep --multi_subject --subj=1 --batch_size=42 --max_lr=3e-4 --mixup_pct=.33 --num_epochs=300 --use_prior --prior_scale=30 --clip_scale=1 --no-blurry_recon --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40
```

### What are the "behav", "past_behav", "future_behav", "old_behav" arrays?

Our webdatasets only contain behavioral information; the brain activations and the seen images get loaded separately from hdf5 files and then indexed from these behav arrays accordingly. The webdataset tar files contain behav/past_behav/future_behav/old_behav matrices, although we only used "behav" for training MindEye2 (the other matrices can still be useful however, so we include them for you.)

Below is the lookup table for these arrays, with variables referenced from the Natural Scenes Dataset manual: https://cvnlab.slite.page/p/fRv4lz5V2F/Untitled

```
0 = COCO IDX (73K) (used to index coco_images_224_float16.hdf5)
1 = SUBJECT
2 = SESSION
3 = RUN
4 = TRIAL
5 = GLOBAL TRIAL (used to index betas_all_subj_fp32_renorm.hdf5)
6 = TIME
7 = ISOLD
8 = ISCORRECT
9 = RT
10 = CHANGEMIND
11 = ISOLDCURRENT
12 = ISCORRECTCURRENT
13 = TOTAL1
14 = TOTAL2
15 = BUTTON
16 = IS_SHARED1000
```

E.g., behav[0,:,9] corresponds to the 1st sample in the current batch's corresponding response time for the participant to press a button for that image.

-1 values in these arrays should be interpreted as NaNs.

past_behav gives you the behavioral information for samples corresponding the immediate previous timepoints samples.

future_behav gives you the behavioral information for samples corresponding to the immediate future timepoint samples.

old_behav gives you the behavioral information for the other repetitions of the given sample (remember to ignore -1s).

The code to create the above webdatasets and the hdf5 full of voxel brain activations can be found in src/dataset_creation.ipynb.


## Citation

If you make use of this work please cite the MindEye2 and MindEye1 papers and the Natural Scenes Dataset paper.

<br>

*MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data*

Scotti, Tripathy, Torrico, Kneeland, Chen, Narang, Santhirasegaran, Xu, Naselaris, Norman, & Abraham. MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data. International Conference on Machine Learning. (2024). arXiv:2403.11207  

<br>

*MindEye1: Reconstructing the Mind's Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors*

Scotti, Banerjee, Goode, Shabalin, Nguyen, Cohen, Dempster, Verlinde, Yundler, Weisberg, Norman, & Abraham. Reconstructing the Mind's Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors. Advances in Neural Information Processing Systems, 36. (2023). arXiv:2305.18274. 

<br>

*Natural Scenes Dataset: A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence*

Allen, St-Yves, Wu, Breedlove, Prince, Dowdle, Nau, Caron, Pestilli, Charest, Hutchinson, Naselaris, & Kay. A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. Nature Neuroscience (2021).
