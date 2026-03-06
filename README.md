# Chattiori's Model Merger

A fast, deterministic **checkpoint merger** for Stable Diffusion–family models (SD1.x/2.x/XL) and **Flux.1** (auto-detected).  
Supports **`.ckpt`** and **`.safetensors`**. Runs on **CPU by default** (GPU optional).

---

## Highlights

- ✅ 2-model and 3-model merges  
- ✅ **Cosine structure modes** (`--cosine0/1/2`) to keep one model’s **structure** while blending **details** from others  
- ✅ Block-weighted / Elemental ratios, plus random ratio samplers  
- ✅ Extra modes: orthogonal delta, sparse top-k, channel-wise cosine gate, frequency-band mix, etc.  
- ✅ VAE bake-in, pruning, EMA keep, fp16/fp8 saving, rich metadata

---

## Architectures & Formats

- **Architectures:** SD 1.x / 2.x / XL / Flux.1 / Z-Image / Anima (via `detect_arch`)
- **Checkpoints:** `.ckpt` (PyTorch) and `.safetensors`
- **VAE:** Optional bake-in with `--vae`
- **DTypes:** fp32 (default), **fp16** (`--save_half`), **fp8** (`--save_quarter`, experimental)

---

## Modes

Pick one `mode` (first positional arg). Some require a third model or `beta`.

| Code   | Name                         | Needs 3rd model | Needs `beta` | Notes |
|-------:|------------------------------|:-------------:|:------------:|------|
| `WS`   | Weighted Sum                 |       ✗       |      ✗       | Standard linear mix |
| `SIG`  | Sigmoid                      |       ✗       |      ✗       | Smooth non-linear |
| `GEO`  | Geometric                    |       ✗       |      ✗       | Geometric mean |
| `MAX`  | Max                          |       ✗       |      ✗       | Element-wise max |
| `AD`   | Add Difference               |       ✓       |      ✗       | Uses (model1 − model2) |
| `sAD`  | Smooth Add Difference        |       ✓       |      ✗       | Median+Gaussian filtered diff |
| `MD`   | Multiply Difference          |       ✓       |      ✓       | Non-linear diff product |
| `SIM`  | Similarity Add Difference    |       ✓       |      ✓       | Similarity-aware lerp |
| `TD`   | Train Difference             |       ✓       |      ✗       | Data-dependent scaling |
| `TRS`  | Triple Sum                   |       ✓       |      ✓       | α/β tri-blend |
| `TS`   | Tensor Sum                   |       ✗       |      ✓       | Splice by ranges |
| `ST`   | Sum Twice                    |       ✓       |      ✓       | Two-stage sum |
| `NoIn` | No Interpolation             |       ✗       |      ✗       | No blending; combine with `--fine` |
| `RM`   | Read Metadata                |       ✗       |      ✗       | Prints SHA256 + metadata and exits |
| `DARE` | DARE                         |       ✗       |      ✓       | Stochastic delta resampling |
| `ORTHO`| Orthogonalized Delta         |       ✗       |      ✗       | Apply diff orthogonal to base |
| `SPRSE`| Sparse Top-k Delta           |   ✓    |      ✗       | Applies largest diffs (uses α) |
| `NORM` | Norm/Direction Split         |       ✗       |      ✗       | Blend magnitude & direction separately |
| `CHAN` | Channel-wise Cosine Gate     |   ✓    |      ✗       | Gate per output channel (Conv/Linear) |
| `FREQ` | Frequency-Band Blend         |   ✓    |      ✗       | Low/high-freq mix for Conv kernels |
| `SWAP` | Swap Component         |   ✗    |      ✗       | Swap the component like CLIP |
| `CLIPXOR` | CLIP XOR Blend         |   ✗    |      ✗       | Extending CLIPs using XOR Blend |
| `XDARE` | XOR CLIP DARE         |   ✗    |      ✓       | DARE Merge + CLIPXOR |
| `FWM` | Feature Weighted Merge         |   ✗    |      ✗       | True merges between different base model, same architecture |

---

## Cosine Structure Modes

Use **exactly one** of: `--cosine0`, `--cosine1`, `--cosine2` (mutually exclusive).

- `--cosine0`: **Model 0** defines the **structure**; inject details from Model 1 (and Model 2 if present)
- `--cosine1`: **Model 1** defines the **structure**; inject details from the others
- `--cosine2`: **Model 2** defines the **structure**; inject details from Model 0 then Model 1 (**requires** `--model_2`)

These modes compute per-layer cosine stats between the **base (structure)** and **detail** models, then blend using your **`alpha`** (detail A) and **`beta`** (detail B) with block/elemental weights.

---

## Ratios & Block Weights

- `--alpha`, `--beta` accept:
  - A **float** (`0.45`, etc.)
  - **Merge Block Weights** string (19/25-length, etc.)
  - **Elemental Merge** syntax
- `--rand_alpha`, `--rand_beta` accept `"MIN, MAX[, SEED][, elemental...]"`  
  If `SEED` omitted, a random seed is generated.
- **XL note:** 25-length weights auto-convert for XL; 19-length is supported.

Rule of thumb: `--alpha 0.5` ≈ 50/50 of model0 & model1 (lower alpha → closer to model0).

---

## CLI

### Positional

```
mode model_path model_0 model_1
```

- `mode`: one of the codes above
- `model_path`: directory containing the checkpoints
- `model_0`: filename of the first model
- `model_1`: filename of the second model (required for merging modes)
- `model_2`: filename of the third model (required by some modes, or when using `--cosine2`)

### Options

- Ratios:
  - `--alpha <v>` / `--rand_alpha "<min,max[,seed][,elemental...]>"`
  - `--beta  <v>` / `--rand_beta  "<min,max[,seed][,elemental...]>"`
- Cosine structure (mutually exclusive):
  - `--cosine0` | `--cosine1` | `--cosine2` (`--cosine2` requires `--model_2`)
- Finetune:
  - `--fine "<comma,separated,values>"`  
    SDXL: classic pattern; Flux.1: pattern-based scaling for keys like `double_block`, `img_in`, `txt_in`, `time`, `out`, etc.
- I/O & formats:
  - `--vae <path>` bake into `first_stage_model.*`
  - `--save_safetensors` (otherwise saves `.ckpt`)
  - `--save_half` (fp16), `--save_quarter` (fp8, experimental)
  - `--output <name>` output filename (no extension), `--force` to overwrite/autoname
  - `--delete_source` delete source checkpoints after saving
  - `--no_metadata` save without metadata
- Pruning:
  - `--prune` (optionally `--keep_ema` to keep EMA only)
- Names:
  - `--m0_name`, `--m1_name`, `--m2_name`
- Device:
  - `--device cpu|cuda:x` (default `cpu`)
- DARE reproducibility:
  - `--seed <int>`
- Differences:
  - `--use_dif_10`, `--use_dif_20`, `--use_dif_21` to reuse model diffs internally

---

## Usage Examples

### 1) Simple weighted sum (2 models)
```bash
python merge.py WS models "A.safetensors" "B.safetensors"   --alpha 0.45 --output merged_ws --save_safetensors --save_half
```

### 2) Cosine structure: keep model1’s structure, inject model0 details
```bash
python merge.py WS models "A.safetensors" "B.safetensors"   --cosine1 --alpha 0.35 --output merged_cos1
```

### 3) Cosine2 (3 models): keep model2’s structure; inject 0 then 1 with α/β
```bash
python merge.py WS models "A.safetensors" "B.safetensors" "C.safetensors"   --cosine2 --alpha 0.25 --beta 0.15 --output merged_cos2
```

### 4) Sparse Top-k delta (current build marks it as needs model_2)
```bash
python merge.py SPRSE models "A.safetensors" "B.safetensors" "C.safetensors"   --alpha 0.5 --output merged_sparse
```

### 5) Frequency-band blend (Conv kernels; marked as needs model_2)
```bash
python merge.py FREQ models "A.safetensors" "B.safetensors" "C.safetensors"   --alpha 0.4 --output merged_freq
```

### 6) No interpolation + finetune only (e.g., Flux.1 quick tweak)
```bash
python merge.py NoIn models "FluxModel.safetensors" --fine "2,0,1,0,0,5" --output flux_finetuned
```

### 7) Read metadata only
```bash
python merge.py RM models "A.safetensors" --output meta_dump
```

---

## VAE, Pruning & DTypes

- **VAE bake:** `--vae path/to/vae.safetensors` replaces `first_stage_model.*`
- **Pruning:** `--prune` (with `--keep_ema`) to slim the checkpoint; architecture-aware
- **DTypes:** `--save_half` (fp16) is widely supported; `--save_quarter` (fp8) is experimental

---

## Output & Metadata

- **`.safetensors`** (recommended) with metadata (unless `--no_metadata`), or **`.ckpt`** (`{"state_dict": ...}`)
- Metadata includes:
  - SHA256s, legacy hashes
  - `sd_merge_recipe` (method, α/β info, fp, VAE, pruning flags, etc.)
  - `sd_merge_models` (per-model provenance)

---

## Colab Quickstart

```bash
pip install torch safetensors

git clone https://github.com/Faildes/merge-models
cd merge-models

python merge.py WS models "A.safetensors" "B.safetensors" --alpha 0.45 --output merged
```

> Depending on your environment, you may need `python3`, `py`, or `conda run -n <env> python`.

---

## Troubleshooting

- **Cosine switches are exclusive:** use one of `--cosine0/1/2`.  
- **3-model requirements:** `AD`, `sAD`, `MD`, `SIM`, `TD`, `TRS`, `ST` (and in this build: `SPRSE`, `CHAN`, `FREQ`) require `--model_2`.  
- **Low VRAM:** stick to `--device cpu` (default). GPU merging needs VRAM ≈ model sizes (safetensors) or ~1.15× (ckpt).  
- **XL block weights:** 25-length auto-converted; 19-length supported.  
- **FREQ mode:** applies only to Conv tensors with kernel ≥ 3×3; others fall back to simple blending.  
- **Flux finetune:** If some keys don’t react, extend the key substrings (`double_block`, `img_in`, `txt_in`, `time`, `out`, etc.) to match your checkpoint.

---

## Credits

- [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) — overall design inspiration  
- [eyriewow](https://github.com/eyriewow/merge-models) — original merge-models  
- [lopho](https://github.com/lopho/stable-diffusion-prune), [arenasys](https://github.com/arenasys/stable-diffusion-webui-model-toolkit) — pruning  
- [idelairre](https://github.com/idelairre/sd-merge-models) — Geometric / Sigmoid / Max  
- [s1dlx](https://github.com/s1dlx/meh) — Multiply Difference / Similarity Add Difference  
- [hako-mikan](https://github.com/hako-mikan/sd-webui-supermerger) — Tensor Sum / Train Difference / Triple Sum / Sum Twice / Smooth Add Difference / Finetune / Cosine / Elemental  
- [bbc-mc](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui) — Block Weighted Merge  
- [martyn](https://github.com/martyn/safetensors-merge-supermario) — DARE  
- wise-ft origin discussion via r_Sh4d0w → [mlfoundations/wise-ft](https://github.com/mlfoundations/wise-ft)
