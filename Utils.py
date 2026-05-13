from __future__ import annotations
import os, shutil
import json
import re
import numpy as np
import random
import torch
import safetensors
import filelock
import hashlib
from tqdm.auto import tqdm
from typing import List, Tuple, NamedTuple
from pathlib import Path
import torch.nn.functional as F
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from functools import lru_cache

try:
    FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)
    FP8_E5M2 = getattr(torch, "float8_e5m2", None)
    FP8_DTYPES = tuple(d for d in (FP8_E4M3, FP8_E5M2) if d is not None)
except Exception:
    FP8_E4M3 = FP8_E5M2 = None
    FP8_DTYPES = ()

FP_SET = {torch.float32, torch.float16, torch.float64, torch.bfloat16}

NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

BLOCKID = ["BASE"] + [f"IN{i:02}" for i in range(12)] + ["M00"] + [f"OUT{i:02}" for i in range(12)]
BLOCKIDXLL = ["BASE"] + [f"IN{i:02}" for i in range(9)] + ["M00"] + [f"OUT{i:02}" for i in range(9)] + ["VAE"]
BLOCKIDFLUX = ["CLIP", "T5", "IN"] + ["D{:002}".format(x) for x in range(19)] + ["S{:002}".format(x) for x in range(38)] + ["OUT"] # Len: 61
BLOCKIDZI = ["BASE","CONT","NOISE"] + [f"L{i:02}" for i in range(30)] + ["VAE"]
BLOCKIDAM = ["BASE"] + [f"L{i:02}" for i in range(28)] + ["VAE"] # Anima Model has 28 blocks

_re_inp = re.compile(r'\.input_blocks\.(\d+)\.')
_re_mid = re.compile(r'\.middle_block\.(\d+)\.')
_re_out = re.compile(r'\.output_blocks\.(\d+)\.')

_re_flux_any_num = re.compile(r'\.(\d+)\.')
_re_xl_tf = re.compile(r'transformer_blocks\.(\d+)\.')

COLS = [[-1, 1/3, 2/3], [1, 1, 0], [0, -1, -1], [1, 0, 1]]
COLSXL = [[0, 0, 1], [1, 0, 0], [-1, -1, 0], [-1, 1, 0]]

BNB = ".quant_state.bitsandbytes__"
QTYPES = ["fp4", "nf4"]

FINETUNES = [
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.input_blocks.0.0.bias",
    "model.diffusion_model.out.0.weight",
    "model.diffusion_model.out.0.bias",
    "model.diffusion_model.out.2.weight",
    "model.diffusion_model.out.2.bias",
]

# ---------------------------------------------------------------------
# LBLOCKS (per-arch)
#  - each entry corresponds 1:1 with that arch's BLOCKID* order
#  - each entry is a list of substrings (aliases) to match against "full" or "msd"
# ---------------------------------------------------------------------

def make_lblocks_sdxl():
    """
    Order aligns with BLOCKIDXLL:
      BASE, IN00..IN08, M00, OUT00..OUT08, VAE
    """
    lblocks = []

    # BASE (text/conditioner)
    lblocks.append([
        "conditioner.embedders.",     # SDXL conditioner
        "text_encoders.",             # diffusers packs
        "clip_l.", "clip_g.",         # alt roots
        "0_transformer_text_model_",  # your convert() output for te1
        "1_model_transformer_resblocks_",  # your convert() output for te2
    ])

    # IN00..IN08
    for i in range(9):
        lblocks.append([
            f"diffusion_model_input_blocks_{i}_",          # your convert() output
            f"model.diffusion_model.input_blocks.{i}.",    # raw checkpoint
        ])

    # M00
    lblocks.append([
        "diffusion_model_middle_block_",
        "model.diffusion_model.middle_block.",
    ])

    # OUT00..OUT08
    for i in range(9):
        lblocks.append([
            f"diffusion_model_output_blocks_{i}_",
            f"model.diffusion_model.output_blocks.{i}.",
        ])

    # SDXL has extra ".out." heads; fold them into OUT08 bucket
    lblocks[-1].extend([
        "diffusion_model_out_",
        "model.diffusion_model.out.",
    ])

    # VAE
    lblocks.append([
        "first_stage_model.",
        "vae.",
    ])

    return lblocks


def make_lblocks_flux():
    """
    Order aligns with BLOCKIDFLUX:
      CLIP, T5, IN, D00..D18, S00..S37, OUT
    """
    lblocks = []

    # CLIP
    lblocks.append([
        "text_encoders.clip", "clip.", "clip_l.", "clip_g.",
        "text_encoder.", "conditioner.embedders.",
    ])

    # T5
    lblocks.append([
        "t5xxl", "t5.", "text_encoders.t5", "text_encoder_2.",
    ])

    # IN (input projections / embeddings)
    lblocks.append([
        "img_in", "txt_in", "time_in", "vector_in",
        "x_embedder", "t_embedder",
    ])

    # D00..D18 (double blocks)
    for i in range(19):
        lblocks.append([f"double_blocks.{i}.", f"double_block.{i}."])

    # S00..S37 (single blocks)
    for i in range(38):
        lblocks.append([f"single_blocks.{i}.", f"single_block.{i}."])

    # OUT (final layer / heads)
    lblocks.append([
        "final_layer", "out.", "vector_out",
    ])

    return lblocks


def make_lblocks_zi():
    """
    Order aligns with BLOCKIDZI:
      BASE, CONT, NOISE, L00..L29, VAE
    """
    lblocks = []

    # BASE (text enc / caption embedder)
    lblocks.append([
        "text_encoders.qwen3_4b.", "qwen3_4b.",
        "model.diffusion_model.cap_embedder.", "cap_embedder.",
    ])

    # CONT / NOISE
    lblocks.append(["context_refiner", "diffusion_model.context_refiner"])
    lblocks.append(["noise_refiner",   "diffusion_model.noise_refiner"])

    # L00..L29
    for i in range(30):
        lblocks.append([
            f"diffusion_model.layers.{i}.",         # LoRA example (no 'model.' prefix)
            f"model.diffusion_model.layers.{i}.",   # model keys
            f"diffusion_model_layers_{i}_",         # if you ever convert to underscore form
        ])

    # VAE
    lblocks.append([
        "vae.",
        "first_stage_model.",
    ])

    return lblocks

def make_lblocks_am():
    """
    Order aligns with BLOCKIDAM:
      BASE, L00..L27, VAE

    Supports checkpoint keys:
      - official:   net.* (then normalized)
      - unofficial: no net.
      - AIO:        cond_stage_model.qwen3_06b.* + model.diffusion_model.* + first_stage_model.*

    Supports LoRA keys (kohya-style):
      - TE:   lora_te_layers_{n}_...
      - UNet: lora_unet_blocks_{n}_...
      - VAE:  lora_vae_* or lora_first_stage_model_*
    """
    lblocks = []

    # -------------------------
    # BASE (text + adapters/embeds)
    # -------------------------
    lblocks.append([
        # ---- AIO text encoder (checkpoint) ----
        "cond_stage_model.qwen3_06b.",

        # ---- canonicalized text encoder (if you normalize AIO -> text_encoders.*) ----
        "text_encoders.qwen3_06b.", "text_encoders.qwen3_06b_base.",
        "qwen3_06b.", "qwen3_06b_base.",

        # ---- diffusion-side adapters / time/pos ----
        "model.diffusion_model.llm_adapter.", "diffusion_model.llm_adapter.", "llm_adapter.",
        "model_diffusion_model_llm_adapter_", "diffusion_model_llm_adapter_", "net_llm_adapter_",
        "model.diffusion_model.t_embedder.", "diffusion_model.t_embedder.", "t_embedder.",
        "model_diffusion_model_t_embedder_", "diffusion_model_t_embedder_", "net_t_embedder_",
        "model.diffusion_model.t_embedding_norm.", "diffusion_model.t_embedding_norm.", "t_embedding_norm.",
        "model.diffusion_model.pos_embedder.", "diffusion_model.pos_embedder.", "pos_embedder.",

        # ---- official may come as net.* ----
        "net.llm_adapter.", "net.t_embedder.", "net.t_embedding_norm.", "net.pos_embedder.",

        # ---- LoRA: TE side (all goes to BASE, like ZI) ----
        "lora_te_",                 # generic
        "lora_te_layers_",          # matches: lora_te_layers_7_...
        "lora_text_encoder_",       # some trainers use this

        # ---- LoRA: sometimes text is named as cond_stage_model in lora packs ----
        "lora_cond_stage_model_",   # just in case
    ])

    # -------------------------
    # L00..L27 (diffusion blocks)
    # -------------------------
    for i in range(28):
        lblocks.append([
            # checkpoint keys
            f"model.diffusion_model.blocks.{i}.",
            f"blocks.{i}.",
            f"net.blocks.{i}.",

            # LoRA keys (kohya-style + direct module-path style)
            f"lora_unet_blocks_{i}_",     # matches: lora_unet_blocks_0_self_attn_q_proj...
            f"lora_unet_blocks.{i}.",     # safety (rare)
            f"diffusion_model.blocks.{i}.",
            f"model.diffusion_model.blocks.{i}.",
            f"net.blocks.{i}.",
            f"diffusion_model_blocks_{i}_",       # convert_diffusers_name_to_compvis() direct-key form
            f"model_diffusion_model_blocks_{i}_",
            f"net_blocks_{i}_",
        ])

    # Add x_embedder into L00 bucket (input-like)
    lblocks[1].extend([
        "model.diffusion_model.x_embedder.", "diffusion_model.x_embedder.", "x_embedder.",
        "net.x_embedder.",
        "model_diffusion_model_x_embedder_", "diffusion_model_x_embedder_", "net_x_embedder_",
        "lora_unet_x_embedder", "lora_unet_x_embedder_",  # just in case
    ])

    # Add final_layer into L27 bucket (output-like)
    lblocks[28].extend([
        "model.diffusion_model.final_layer.", "diffusion_model.final_layer.", "final_layer.",
        "net.final_layer.",
        "model_diffusion_model_final_layer_", "diffusion_model_final_layer_", "net_final_layer_",
        "lora_unet_final_layer", "lora_unet_final_layer_",  # just in case
        "lora_unet_out", "lora_unet_out_",                  # some packs use out naming
    ])

    # -------------------------
    # VAE
    # -------------------------
    lblocks.append([
        "vae.",
        "first_stage_model.",

        # LoRA (if present)
        "lora_vae_", "lora_first_stage_model_", "lora_fs_", "lora_vae",
    ])

    return lblocks


# Convenient ready-to-use constants
LBLOCKS_SDXL = make_lblocks_sdxl()
LBLOCKS_FLUX = make_lblocks_flux()
LBLOCKS_ZI   = make_lblocks_zi()
LBLOCKS_AM = make_lblocks_am()

LBLOCKS26 = [
    "encoder",
    "diffusion_model_input_blocks_0_","diffusion_model_input_blocks_1_","diffusion_model_input_blocks_2_",
    "diffusion_model_input_blocks_3_","diffusion_model_input_blocks_4_","diffusion_model_input_blocks_5_",
    "diffusion_model_input_blocks_6_","diffusion_model_input_blocks_7_","diffusion_model_input_blocks_8_",
    "diffusion_model_input_blocks_9_","diffusion_model_input_blocks_10_","diffusion_model_input_blocks_11_",
    "diffusion_model_middle_block_",
    "diffusion_model_output_blocks_0_","diffusion_model_output_blocks_1_","diffusion_model_output_blocks_2_",
    "diffusion_model_output_blocks_3_","diffusion_model_output_blocks_4_","diffusion_model_output_blocks_5_",
    "diffusion_model_output_blocks_6_","diffusion_model_output_blocks_7_","diffusion_model_output_blocks_8_",
    "diffusion_model_output_blocks_9_","diffusion_model_output_blocks_10_","diffusion_model_output_blocks_11_",
    "embedders",
]

checkpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}

checkpoint_dict_skip_on_merge = set(["cond_stage_model.transformer.text_model.embeddings.position_ids"])

def normalize_path(path: str) -> str:
    path = os.path.abspath(path)
    if os.name == "nt":
        if not path.startswith("\\\\?\\"):
            path = "\\\\?\\" + path
    return str(Path(path).expanduser().resolve())

def tagdict(presets: str) -> dict:
    """Parse presets text into a dict if value part has exactly 26 items."""
    wdict = {}
    for line in presets.splitlines():
        parts = re.split(r'[:\t]', line, maxsplit=1)
        if len(parts) == 2:
            key, w = parts
            if len(w.split(",")) == 26:
                wdict[key.strip()] = w.strip()
    return wdict

def base_path(path: str=None) -> str:
    if path is None:
        return os.path.dirname(os.path.realpath(__file__))
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), path)

file_path = base_path("mbwpresets.txt")
if not os.path.isfile(file_path):
    shutil.copyfile(base_path("mbwpresets_master.txt"), file_path)
weights_presets_list = tagdict(open(file_path).read())

_SPLIT = re.compile(r"[,\n]+")

def _split(s:str):
    return [t.strip() for t in _SPLIT.split(s) if t.strip()]

def _get_cast(xs, i, cast, default):
    try: return cast(xs[i])
    except (IndexError, ValueError): return default

def wgt(x, dp):
    useblocks = False
    if isinstance(x, (int, float)):return float(x), dp, useblocks
    useblocks = True
    nums, rest = deepblock(x if isinstance(x, list) else [x])
    return (nums[0] if len(nums) == 1 else nums), rest, useblocks

def deepblock(items:List[str])->Tuple[List[float],List[str]]:
    nums:List[float]=[];rest:List[str]=[];stack=list(items)
    while stack:
        s = stack.pop()
        src = weights_presets_list.get(s, s)
        for t in _split(src):
            if t in weights_presets_list: stack.append(t); continue
            try: nums.append(float(t))
            except ValueError: rest.append(t)
    return nums, rest

def rinfo(s:str,seed:int)->str:
    core, _, rest=s.replace(" ", "").partition("[")
    fe = rest[:-1] if rest.endswith("]") else None
    toks = _split(core)
    rmin = _get_cast(toks, 0, float, 0.0)
    rmax = _get_cast(toks, 1, float, 1.0)
    get =  _get_cast(toks, 2, int,  seed)
    return f"({rmin},{rmax},{get},[{fe}])"

def roundeep(term):
    if not term: return None
    out=[]
    for d in term:
        try:
            a, b, c = d.split(":", 2)
            out.append(f"{a}:{b}:{round(float(c), 3)}")
        except ValueError: out.append(d)
    return out

def rand_ratio(s:str):
    core, _, rest = s.partition("[")
    deep = _split(rest[:-1]) if rest.endswith("]") else []
    toks = _split(core.replace(" ",""))
    rmin = _get_cast(toks, 0, float, 0.0)
    rmax = _get_cast(toks, 1, float, 1.0)
    seed = _get_cast(toks, 2, int,   random.randint(1, 4294967295))

    np.random.seed(seed)
    ratios = np.random.uniform(rmin, rmax, 26).tolist()
    deep_res = []

    for d in deep:
        if "PRESET" in d:
            try:
                _, pack = d.split(":",1)
                name, drat_s = pack.split("(")
                base_vals = [float(x) for x in _split(weights_presets_list[name])]
                drat = float(drat_s.rstrip(")"))
                ratios = [r * (1 - drat) + b * drat for r, b in zip(ratios, base_vals)]
            except Exception:
                pass
            continue

        if d.count(":") != 2: continue
        dbs_s, dws, dr_s = d.split(":", 2)
        dbs = dbs_s.split()
        if "(" in dr_s:
            v, drat_s = dr_s.split("(")
            v = float(v)
            drat = float(drat_s.rstrip(")"))
            if dws == "ALL":
                for db in dbs:
                    i = BLOCKID.index(db)
                    ratios[i] = ratios[i] * (1 - drat) + v * drat
            else:
                for db in dbs:
                    cur = ratios[BLOCKID.index(db)]
                    deep_res.append(f"{db}:{dws}:{cur * (1 - drat) + v * drat}")
        else:
            v = float(dr_s)
            if dws == "ALL":
                for db in dbs: ratios[BLOCKID.index(db)] = v
            else:
                for db in dbs: deep_res.append(f"{db}:{dws}:{v}")

    info = rinfo(core, seed)
    ratios, deep_res = wgt(ratios, deep_res)
    return ratios, seed, deep_res, info

def colorcalc(cols, isxl):
    M = COLSXL if isxl else COLS
    return [0.02 * sum(v * cols[i] for i, v in enumerate(col)) for col in zip(*M)]

def fineman(fine, arch):
    if arch.get("FLUX", False):
        mul = {
            "double_block": 1.0 + (fine[0] * 0.01) if len(fine) > 0 else 1.0,
            "img_in":       1.0 + (fine[1] * 0.01) if len(fine) > 1 else 1.0,
            "txt_in":       1.0 + (fine[2] * 0.01) if len(fine) > 2 else 1.0,
            "time":         1.0 + (fine[3] * 0.01) if len(fine) > 3 else 1.0,
            "out":          1.0 + (fine[4] * 0.01) if len(fine) > 4 else 1.0,
        }
        add = (fine[5] * 0.02) if len(fine) > 5 else 0.0
        return {"mul": mul, "add": add}
    
    if isinstance(fine, (list, tuple, np.ndarray)) and len(fine) >= 8:
        return {
            "mode": "image_tone_v2",
            "noise1": float(fine[0]),
            "noise2": float(fine[1]),
            "noise3": float(fine[2]),
            "contrast": float(fine[3]),
            "brightness": float(fine[4]),
            "r": float(fine[5]),
            "g": float(fine[6]),
            "b": float(fine[7]),
            # Optional 9th value: direct saturation factor.
            # 1.0 = off, 0.75 = 25% desaturate, 1.15 = boost.
            "saturation": float(fine[8]) if len(fine) > 8 else 1.0,
        }

    r = [
        1 - fine[0] * 0.01,
        1 + fine[0] * 0.02,
        1 - fine[1] * 0.01,
        1 + fine[1] * 0.02,
        1 - fine[2] * 0.01,
        [fine[3] * 0.02] + colorcalc(fine[4:8], arch.get("XL", False))
        ]
    return r

def weighttoxl(weight):
    return weight[:9] + weight[12:22] +[0]

def parse_ratio(ratios, info, dp):
    if isinstance(ratios, list):
        ratio, *weights = ratios
        rounded = [round(a, 3) for a in weights]
        round_deep = roundeep(dp)
        prefix = f"preset:[{info}]," if info else ""
        info = f"{prefix}{round(ratio,3)},[{rounded},[{round_deep}]]"
    else:
        ratio, weights, info = ratios, [ratios]*34, f"{round(ratios,3)}"
    return weights, ratio, info


DTYPES = {torch.float32, torch.float64, torch.bfloat16}

def to_half(tensor, enable):
    return tensor.half() if enable and getattr(tensor, "dtype", None) in DTYPES else tensor


def upcast_fp8_state_dict(theta: dict, target_dtype: torch.dtype = torch.float16):
    if not FP8_DTYPES:
        return theta

    fp8_keys = []
    for k, v in theta.items():
        if isinstance(v, torch.Tensor) and v.dtype in FP8_DTYPES:
            fp8_keys.append(k)

    if not fp8_keys:
        return theta

    for k in tqdm(fp8_keys, total=len(fp8_keys), desc=f"Upcasting {len(fp8_keys)} fp8 tensors..."):
        theta[k] = theta[k].to(target_dtype)

    return theta


@torch.inference_mode()
def prepare_state_dict_for_save(
    theta: dict,
    args,
    *,
    arch: dict,
    vae_prefix: str | None,
    prune: bool = True,
    make_cpu: bool = True,
    make_contiguous: bool = True,
):
    # roots
    if arch.get("FLUX", False):
        cond_prefixes = ("clip.cond_stage_model.",)
    elif arch.get("XL", False):
        cond_prefixes = ("conditioner.",)
    elif arch.get("ZI", False):
        cond_prefixes = ("text_encoders.qwen3_4b.",)
    elif arch.get("AM", False):
        cond_prefixes = (
            "text_encoders.qwen3_06b.",
            "text_encoders.qwen3_06b_base.",
            "cond_stage_model.qwen3_06b.",
            "qwen3_06b.", "qwen3_06b_base.",
        )
    else:
        cond_prefixes = ("cond_stage_model.",)

    roots = (
        "model.diffusion_model.",
        "depth_model.",
        "first_stage_model.",
        "vae.",
        *cond_prefixes,
    )

    want_fp8  = bool(getattr(args, "save_quarter", False)) and bool(FP8_DTYPES)
    want_fp16 = bool(getattr(args, "save_half", False)) and not want_fp8
    want_bf16 = bool(getattr(args, "save_bhalf", False)) and not want_fp8

    fp8_dtype = FP8_E4M3 if FP8_E4M3 is not None else (FP8_E5M2 if FP8_E5M2 is not None else None)
    if want_fp8 and fp8_dtype is None:
        want_fp8 = False

    cpu = torch.device("cpu")
    
    key_list = list(theta.keys())

    for k in tqdm(key_list, desc="Preparing merged model for save...", unit="param"):
        v = theta.get(k, None)

        # prune
        if prune and ((".scale" in k and arch.get("ZI", False)) or ("pos_embedder" in k and arch.get("AM", False)) or (not any(k.startswith(r) for r in roots))):
            theta.pop(k, None)
            continue

        if not isinstance(v, torch.Tensor):
            continue

        # keep_ema
        if getattr(args, "keep_ema", False):
            k_ema = "model_ema." + k[6:].replace(".", "")
            v_ema = theta.get(k_ema, None)
            if isinstance(v_ema, torch.Tensor):
                v = v_ema

        # dtype
        is_vae = bool(vae_prefix) and k.startswith(vae_prefix)
        if (not is_vae) and v.is_floating_point():
            dt = v.dtype
            if want_fp8:
                if dt in FP_SET:
                    v = v.to(fp8_dtype)
            elif want_fp16:
                if dt in {torch.float32, torch.float64, torch.bfloat16, *FP8_DTYPES}:
                    v = v.to(torch.float16)
            elif want_bf16:
                if dt in {torch.float32, torch.float64, torch.float16, *FP8_DTYPES}:
                    v = v.to(torch.bfloat16)
            else:
                if dt in {torch.float16, torch.float64, torch.bfloat16, *FP8_DTYPES}:
                    v = v.to(torch.float32)

        v = v.detach()

        if make_cpu and v.device.type != "cpu":
            v = v.to(cpu, non_blocking=False)
        if make_contiguous and not v.is_contiguous():
            v = v.contiguous()

        theta[k] = v

    return theta

def _safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

def merge_cache_json(cache_path, model_path):
    model_cache_path = os.path.join(model_path, "cache.json")
    base = _safe_load_json(cache_path if os.path.exists(cache_path) else model_cache_path)
    update = _safe_load_json(model_cache_path) if os.path.exists(model_cache_path) else {}

    if isinstance(base, dict) and isinstance(update, dict):
        base.update(update)
    elif isinstance(base, list) and isinstance(update, list):
        base.extend(update)

    with filelock.FileLock(f"{cache_path}.lock"):
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(base, f, ensure_ascii=False, indent=2)
    
def dump_cache(cache_data, cache_path=None):
    with filelock.FileLock(f"{cache_path}.lock"):
        with open(cache_path, "w", encoding="utf8") as f:
            json.dump(cache_data or {}, f, indent=4)

def cache(subsection, cache_data, cache_path=None):
    if cache_data is None:
        with filelock.FileLock(f"{cache_path}.lock"):
            cache_data = _safe_load_json(cache_path)
    if subsection not in cache_data or not isinstance(cache_data.get(subsection), dict):
        cache_data[subsection] = {}
        dump_cache(cache_data, cache_path)
    return cache_data

def model_hash(filename: str) -> str:
    try:
        with open(filename, "rb") as f:
            f.seek(0x100000)
            return hashlib.sha256(f.read(0x10000)).hexdigest()[:8]
    except FileNotFoundError:
        return "NOFILE"

def sha256_from_cache(filename: str, title: str, cache_data, cache_path=None):
    cache_data = cache("hashes", cache_data, cache_path)
    hsect = cache_data.get("hashes", {})
    h = hsect.get(title)
    if not h:
        return None, None, cache_data

    cur_mtime = os.path.getmtime(filename) if os.path.exists(filename) else None
    if h.get("mtime") != cur_mtime:
        return None, None, cache_data

    return h.get("sha256"), h.get("model_hash"), cache_data

def calculate_sha256(filename: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in tqdm(iter(lambda: f.read(chunk_size), b""), desc="Hashing", unit="MB", unit_scale=chunk_size / (1024 * 1024)):
            hasher.update(chunk)
    return hasher.hexdigest()

def sha256(filename: str, title: str, cache_path: str | None) -> str:
    cache_data = cache("hashes", None, cache_path)
    s256_cached, _, cache_data = sha256_from_cache(filename, title, cache_data)
    if s256_cached:
        return s256_cached

    print(f"Calculating sha256 for {filename}: ", end="")
    sha_val = calculate_sha256(filename)
    mhash = model_hash(filename)
    print(sha_val)

    cache_data["hashes"][title] = {
        "mtime": os.path.getmtime(filename) if os.path.exists(filename) else None,
        "sha256": sha_val,
        "model_hash": mhash,
    }
    dump_cache(cache_data, cache_path)
    return sha_val

def read_metadata_from_safetensors(filename):
    with open(filename, "rb") as f:
        size = int.from_bytes(f.read(8), "little")
        start = f.read(2)
        assert size > 2 and start in (b'{"', b"{'"), f"{filename} is not a safetensors file"
        meta = json.loads(start + f.read(size - 2))

    res = {}
    for k, v in meta.get("__metadata__", {}).items():
        if isinstance(v, str) and v.startswith("{"):
            try: v = json.loads(v)
            except: pass
        res[k] = v
    return res


def qdtyper(sd):
    if any("fp4" in k for k in sd): return "fp4"
    if any("nf4" in k for k in sd): return "nf4"
    for v in sd.values():
        dt = getattr(v, "dtype", None)
        if dt is not None: return dt
        
def to_qdtype(sd1, sd2, qd1, qd2, device):
    t1 = t2 = torch.float16 if (qd1 in QTYPES and qd2 in QTYPES) else None
    if qd1 in QTYPES: sd1, _ = q_dequantize(sd1, qd1, device, t1)
    if qd2 in QTYPES: sd2, _ = q_dequantize(sd2, qd2, device, t2)
    return sd1, sd2

def maybe_to_qdtype(a, b, qa, qb, device, isflux: bool = True):
    return to_qdtype(a, b, qa, qb, device) if isflux and qa != qb else (a, b)

def detect_arch(theta):
    keys = list(theta.keys())

    # -------------------------
    # Anima (AM) detection
    # -------------------------
    def _is_am_key(k: str) -> bool:
        # diffusion signature (DiT blocks + adaln + proj names)
        if (".blocks." in k or k.startswith(("blocks.", "net.blocks.", "model.diffusion_model.blocks."))):
            if ("adaln_modulation_" in k) or (".q_proj.weight" in k) or (".k_proj.weight" in k) or (".v_proj.weight" in k):
                return True
            if (".self_attn." in k) or (".cross_attn." in k):
                return True
        # diffusion head / adapters
        if (
            "final_layer.linear.weight" in k
            or "final_layer.adaln_modulation" in k
            or "llm_adapter." in k
            or "t_embedding_norm" in k
            or (k.startswith(("diffusion_model.final_layer.", "model.diffusion_model.final_layer.", "net.final_layer.", "final_layer.")) and ".lora_" in k)
            or (k.startswith(("diffusion_model.llm_adapter.", "model.diffusion_model.llm_adapter.", "net.llm_adapter.", "llm_adapter.")) and ".lora_" in k)
        ):
            return True
        # AIO text encoder signature
        if k.startswith("cond_stage_model.qwen3_06b."):
            return True
        return False

    isam = any(_is_am_key(k) for k in keys)

    # -------------------------
    # Z-Image (ZI) detection (avoid x_embedder/t_embedder false positives)
    # -------------------------
    iszi = (not isam) and (
        ("model.diffusion_model.cap_embedder.0.weight" in theta) or
        ("cap_embedder.0.weight" in theta) or
        any(k.startswith(("model.diffusion_model.layers.", "diffusion_model.layers.", "layers.")) for k in keys) or
        any("context_refiner" in k for k in keys) or
        any("noise_refiner" in k for k in keys)
    )

    isxl = ("conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in theta)

    # Flux detection (keep both keys for compatibility)
    isflux = any(("double_blocks" in k) or ("double_block" in k) or ("single_blocks" in k) or ("single_block" in k) for k in keys)

    arch = {
        "XL":   bool(isxl),
        "FLUX": bool(isflux),
        "ZI":   bool(iszi),
        "AM":   bool(isam),
    }

    # -------------------------
    # normalize keys for ZI
    # -------------------------
    if iszi:
        def _zi_key(k: str) -> str:
            if k.startswith(("vae.", "text_encoders.", "model.diffusion_model.")):
                return k

            if k.startswith("diffusion_model."):
                return "model." + k

            if k.startswith(("layers.", "context_refiner.", "noise_refiner.", "final_layer.", "cap_embedder.")):
                return "model.diffusion_model." + k

            return k

        theta = {_zi_key(k): v for k, v in theta.items()}

    # -------------------------
    # normalize keys for AM (format conversion)
    #   official:   net.* -> model.diffusion_model.*
    #   unofficial: blocks.* -> model.diffusion_model.blocks.*
    #   AIO text:   cond_stage_model.qwen3_06b.* -> text_encoders.qwen3_06b.*
    # -------------------------
    if isam:
        def _am_key(k: str) -> str:
            # VAE keep
            if k.startswith(("vae.", "first_stage_model.")):
                return k

            # AIO text encoder -> canonical text_encoders.*
            if k.startswith("text_encoders.qwen3_06b."):
                return "cond_stage_model.qwen3_06b." + k[len("text_encoders.qwen3_06b."):]

            # already canonical diffusion
            if k.startswith("model.diffusion_model."):
                return k

            # strip net. if present
            k2 = k[4:] if k.startswith("net.") else k

            # diffusion modules: map to model.diffusion_model.*
            if k2.startswith((
                "blocks.",
                "x_embedder.",
                "t_embedder.",
                "t_embedding_norm.",
                "pos_embedder.",
                "final_layer.",
                "llm_adapter.",
            )):
                return "model.diffusion_model." + k2

            # some packs may use diffusion_model.* without model.
            if k2.startswith("diffusion_model."):
                return "model." + k2

            # split-file style text encoders already ok
            if k2.startswith(("cond_stage_model.", "qwen3_")):
                return k2

            return k

        theta = {_am_key(k): v for k, v in theta.items()}

    return arch, theta


def _common_dtype(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    dt = torch.promote_types(torch.promote_types(a.dtype, b.dtype), c.dtype)
    if dt in FP8_DTYPES:
        dt = torch.float16
    return dt


def q_dequantize(sd, qtype, device, dtype, setbnb=True):
    from bitsandbytes.functional import dequantize_4bit
    dels = []
    calc = "cuda:0" if torch.cuda.is_available() else ("mps:0" if torch.backends.mps.is_available() else "cpu")
    for k, v in list(sd.items()):
        qk = k + BNB + qtype
        if ("weight" in k) and ("weight." not in k) and (qk in sd):
            qs   = q_tensor_to_dict(sd[qk])
            out  = torch.empty(qs["shape"], device = calc)
            deq  = dequantize_4bit(v.to(calc), out = out,
                                   absmax = sd[k + ".absmax"].to(calc),
                                   blocksize = qs["blocksize"], quant_type = qs["quant_type"])
            sd[k] = deq.to(device, dtype) if dtype else deq.to(device)
            dels += [k + ".absmax", k + ".quant_map"] + ([qk] if setbnb else [])
        elif isinstance(v, torch.Tensor) and dtype:
            sd[k] = v.to(dtype)
    for k in dels: sd.pop(k, None)
    return sd, dtype

def q_tensor_to_dict(t):
    return json.loads(bytes(t.tolist()).decode("utf-8"))

_blocker_cache = {}

def blocker(blocks: str, blockids: list[str]) -> str:
    ck = (id(blockids), blocks)
    hit = _blocker_cache.get(ck)
    if hit is not None:
        return hit

    out = []
    idx = {b:i for i,b in enumerate(blockids)}
    for w in blocks.split():
        if "-" in w:
            a, b = (t.strip() for t in w.split("-", 1))
            i, j = idx[a], idx[b]
            lo, hi = (i, j) if i <= j else (j, i)
            out.extend(blockids[lo:hi + 1])
        else:
            out.append(w)

    res = " ".join(out)
    _blocker_cache[ck] = res
    return res

def _parse_int_after(s: str, marker: str):
    i = s.find(marker)
    if i < 0:
        return None
    j = i + len(marker)
    k = j
    while k < len(s) and s[k].isdigit():
        k += 1
    if k == j:
        return None
    try:
        return int(s[j:k])
    except ValueError:
        return None

def _digits_concat(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())

@lru_cache(maxsize=250_000)
def _blockfromkey_cached_flags(key: str, xl: bool, flux: bool, zi: bool, am: bool) -> Tuple[str, str]:
    arch = {"XL": bool(xl), "FLUX": bool(flux), "ZI": bool(zi), "AM": bool(am)}

    # -------------------------
    # SD1.5 / SD2.x (non-XL/non-Flux/non-ZI/non-AM)
    # -------------------------
    if not (arch["XL"] or arch["FLUX"] or arch["ZI"] or arch["AM"]):
        if "time_embed" in key:
            idx = -2
        elif ".out." in key:
            idx = NUM_TOTAL_BLOCKS - 1
        else:
            m = _re_inp.search(key)
            if m:
                idx = int(m.group(1))
            else:
                if _re_mid.search(key):
                    idx = NUM_INPUT_BLOCKS
                else:
                    m2 = _re_out.search(key)
                    if m2:
                        idx = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + int(m2.group(1))
                    else:
                        return "Not Merge", "Not Merge"

        b = BLOCKID[idx + 1]
        return b, b

    # -------------------------
    # Flux
    # -------------------------
    if arch.get("FLUX", False):
        if "vae" in key:
            return "VAE", "Not Merge"
        if "t5xxl" in key:
            return "T5", "T5"
        if "text_encoders.clip" in key:
            return "CLIP", "CLIP"

        di = _parse_int_after(key, "double_blocks.")
        if di is None:
            di = _parse_int_after(key, "double_block.")
        if di is not None:
            tag = f"D{di:02d}"
            return tag, tag

        si = _parse_int_after(key, "single_blocks.")
        if si is None:
            si = _parse_int_after(key, "single_block.")
        if si is not None:
            tag = f"S{si:02d}"
            return tag, tag

        if "_in" in key:
            return "IN", "IN"
        if "final_layer" in key:
            return "OUT", "OUT"

        m = _re_flux_any_num.search(key)
        if m and "double_blocks" in key:
            tag = f"D{int(m.group(1)):02d}"
            return tag, tag
        if m and "single_blocks" in key:
            tag = f"S{int(m.group(1)):02d}"
            return tag, tag

        return "Not Merge", "Not Merge"

    # -------------------------
    # SDXL
    # -------------------------
    if arch.get("XL", False):
        if not ("weight" in key or "bias" in key):
            return "Not Merge", "Not Merge"
        if "label_emb" in key or "time_embed" in key:
            return "Not Merge", "Not Merge"
        if "conditioner.embedders" in key:
            return "BASE", "BASE"

        if "first_stage_model" in key:
            return "VAE", "BASE"

        if "model.diffusion_model.out." in key:
            return "OUT8", "OUT08"

        if "model.diffusion_model" in key:
            if ".input_blocks." in key:
                blk = "IN"
            elif ".middle_block." in key:
                blk = "MID"
            elif ".output_blocks." in key:
                blk = "OUT"
            else:
                return "Not Merge", "Not Merge"

            nums = _digits_concat(key)
            if not nums:
                return "Not Merge", "Not Merge"

            tag = (nums[:1] + "0") if blk == "MID" else nums[:2]

            add = ""
            mi = _parse_int_after(key, "transformer_blocks.")
            if mi is None:
                m = _re_xl_tf.search(key)
                if m:
                    mi = int(m.group(1))
            if mi is not None:
                add = str(mi)

            left = blk + tag + add
            right = ("M00" if blk == "MID" else f"{blk}0{tag[0]}")
            return left, right

        return "Not Merge", "Not Merge"

    # -------------------------
    # Anima (AM)
    # -------------------------
    if arch.get("AM", False):
        if ("qwen3_06b" in key) or key.startswith("text_encoders.") or key.startswith("cond_stage_model.qwen3_06b."):
            return "BASE", "BASE"

        if "vae" in key or key.startswith("first_stage_model."):
            return "VAE", "VAE"

        li = _parse_int_after(key, "blocks.")
        if li is not None and 0 <= li < 28:
            tag = f"L{li:02d}"
            return tag, tag

        if "x_embedder" in key:
            return "L00", "L00"
        if "final_layer" in key:
            return "L27", "L27"
        if any(s in key for s in ("t_embedder", "t_embedding_norm", "pos_embedder", "llm_adapter")):
            return "BASE", "BASE"

        return "Not Merge", "Not Merge"

    # -------------------------
    # Z-Image
    # -------------------------
    if arch.get("ZI", False):
        if ("qwen3_4b" in key) or ("cap_embedder" in key):
            return "BASE", "BASE"
        if not ("weight" in key or "bias" in key):
            return "Not Merge", "Not Merge"
        if "t_embedder" in key or "x_embedder" in key:
            return "Not Merge", "Not Merge"
        if "vae" in key:
            return "VAE", "VAE"
        if "norm_final" in key:
            return "L29", "L29"

        if "model.diffusion_model" in key:
            if "model.diffusion_model.final_layer" in key:
                return "L29", "L29"
            if "model.diffusion_model.context_refiner" in key:
                return "CONT", "CONT"
            if "model.diffusion_model.noise_refiner" in key:
                return "NOISE", "NOISE"

            li = _parse_int_after(key, "layers.")
            if li is not None:
                tag = f"L{li:02d}"
                return tag, tag

        return "Not Merge", "Not Merge"

    return "Not Merge", "Not Merge"


def blockfromkey(key: str, arch: dict) -> Tuple[str, str]:
    xl   = bool(arch.get("XL", False))
    flux = bool(arch.get("FLUX", False) or arch.get("Flux", False))
    zi   = bool(arch.get("ZI", False))
    am   = bool(arch.get("AM", False))
    return _blockfromkey_cached_flags(key, xl, flux, zi, am)

EXTRA_ELEM_TAGS = ("LABEL", "TIME", "OUT", "CLIP", "CLIP-L", "CLIP-G", "T5")

def extra_tag_for_key(key: str, *, arch: dict) -> str | None:
    kl = key.lower()

    # --- explicit pseudo blocks ---
    if "label_emb" in kl:
        return "LABEL"
    if "time_embed" in kl:
        return "TIME"
    if ".out." in kl or "final_layer" in kl:
        return "OUT"

    # --- CLIP grouping (for elementals targeting) ---
    if arch.get("XL", False):
        # SDXL: CLIP-L / CLIP-G
        if ("conditioner.embedders.0." in kl) or ("text_encoders.encoder_l." in kl) or ("clip_l." in kl):
            return "CLIP-L"
        if ("conditioner.embedders.1." in kl) or ("text_encoders.encoder_g." in kl) or ("clip_g." in kl):
            return "CLIP-G"
        if ("conditioner.embedders." in kl) or ("text_encoders." in kl) or ("clip_l." in kl) or ("clip_g." in kl):
            return "CLIP"

    elif arch.get("FLUX", False):
        # Flux: CLIP / T5
        if ("t5" in kl) or ("t5xxl" in kl) or ("text_encoders.t5" in kl):
            return "T5"
        if ("clip" in kl) or ("text_encoder" in kl) or ("conditioner.embedders" in kl):
            return "CLIP"

    elif arch.get("ZI", False):
        # ZI: Qwen + cap_embedder (treat as CLIP tag for elementals convenience)
        if ("qwen3_4b" in kl) or ("cap_embedder" in kl) or ("text_encoders.qwen3_4b" in kl):
            return "CLIP"
        
    elif arch.get("AM", False):
        if ("qwen3_06b" in kl) or ("text_encoders.qwen3_06b" in kl) or ("cond_stage_model.qwen3_06b" in kl) or ("llm_adapter" in kl):
            return "CLIP"

    else:
        # SD1.x/2.x: cond_stage_model / clip
        if kl.startswith("cond_stage_model.") or kl.startswith("clip."):
            return "CLIP"

    return None


def _extend_blockids_for_elementals(blockids: list[str]) -> list[str]:
    out = list(blockids)
    for t in EXTRA_ELEM_TAGS:
        if t not in out:
            out.append(t)
    return out


def elementals2(
    key: str,
    weight_index: int,
    deep: list[str],
    current_alpha: float,
    *,
    blockids=BLOCKID,
    arch: dict,
) -> float:
    """
    elementals() compatible, but:
      - works even when weight_index < 0 ("Not Merge") if extra_tag_for_key() returns a pseudo tag.
      - appends BOTH (block tag) and (extra tag) to the match key, so deep rules can target:
          CLIP / CLIP-L / CLIP-G / LABEL / TIME / OUT
    """
    if not deep:
        return current_alpha

    block_tag = blockids[weight_index] if (0 <= weight_index < len(blockids)) else ""
    extra_tag = extra_tag_for_key(key, arch=arch)

    # matching string (case-sensitive tags are uppercase; keys are usually lowercase)
    skey = f"{key}|{block_tag}|{extra_tag or ''}"

    blockids_ex = _extend_blockids_for_elementals(list(blockids))

    def _neg(tokens: list[str]):
        return (True, tokens[1:]) if tokens and tokens[0] == "NOT" else (False, tokens)

    for d in deep:
        if d.count(":") != 2:
            continue
        dbs_s, dws_s, dr_s = d.split(":", 2)

        # allow ranges for normal blocks; LABEL/TIME/OUT/CLIP are also accepted
        dbs = blocker(dbs_s, blockids_ex).split()
        dws = dws_s.split()
        dbn, dbs = _neg(dbs)
        dwn, dws = _neg(dws)

        ok = (any(db in skey for db in dbs) ^ dbn)
        if ok:
            ok = (any(dw in skey for dw in dws) ^ dwn)
        if ok:
            current_alpha = float(dr_s)

    return current_alpha

def diff_inplace(dst, src, func, desc):
    for k in tqdm(dst.keys(), desc=desc, total=len(dst)):
        if ("model" not in k) and ("text_encoders" not in k):
            continue
        v2 = src.get(k)
        if v2 is None:
            dst[k] = torch.zeros_like(dst[k])
        else:
            dst[k] = func(dst[k], v2)


def _normalize_components_list(alpha_text: str):
    # "UNet, CLIP-L, VAE" -> {'unet','clip-l','vae'}
    if not alpha_text:
        return set()

    tokens = [t.strip().lower() for t in alpha_text.replace(";", ",").split(",") if t.strip()]
    syn = {
        "u": "unet", "unet": "unet",
        "v": "vae", "vae": "vae",
        "clip": "clip", "text": "clip", "te": "clip",
        "clip-l": "clip-l", "clipl": "clip-l", "clip_l": "clip-l", "l": "clip-l", "text-l": "clip-l",
        "clip-g": "clip-g", "clipg": "clip-g", "clip_g": "clip-g", "g": "clip-g", "text-g": "clip-g",
        "denoiser": "transformer", "denoise": "transformer", "transformer": "transformer", "mmdit": "transformer",
        "t5": "text", "t5-xxl": "text", "text1": "text", "text2": "text2",
        "all": "all",
    }

    mapped = []
    for t in tokens:
        tt = syn.get(t, None)
        if tt is None:
            try:
                float(t)
                continue
            except Exception:
                continue
        mapped.append(tt)

    if (not mapped) or ("all" in mapped):
        return {"unet", "vae", "clip-l", "clip-g", "clip", "transformer", "text", "text2"}

    return set(mapped)

def _parse_components_with_only(alpha_text: str):
    if not alpha_text:
        return set(), set()

    tokens = [t.strip().lower() for t in alpha_text.replace(";", ",").split(",") if t.strip()]

    syn = {
        "u": "unet", "unet": "unet",
        "v": "vae", "vae": "vae",
        "clip": "clip", "text": "clip", "te": "clip",
        "clip-l": "clip-l", "clipl": "clip-l", "clip_l": "clip-l", "l": "clip-l", "text-l": "clip-l",
        "clip-g": "clip-g", "clipg": "clip-g", "clip_g": "clip-g", "g": "clip-g", "text-g": "clip-g",
        "denoiser":"transformer", "denoise":"transformer", "transformer":"transformer", "mmdit":"transformer",
        "t5":"text", "t5-xxl":"text", "text1":"text", "text2":"text2",
        "all": "all",
    }

    all_set = {"unet", "vae", "clip-l", "clip-g", "clip", "transformer", "text", "text2"}

    comps = set()
    only  = set()

    for t in tokens:
        is_only = False
        if t.endswith("_only"):
            is_only = True
            t = t[:-5]

        try:
            float(t)
            continue
        except Exception:
            pass

        t = syn.get(t, None)
        if t is None:
            continue

        if t == "all":
            return set(all_set), set()

        if t in all_set:
            comps.add(t)
            if is_only:
                only.add(t)

    return comps, only

def _component_prefix_map(arch: dict) -> dict[str, list[str]]:
    if arch.get("FLUX", False):
        return {
            "transformer": ["transformer."],
            "vae":         ["vae.", "first_stage_model."],
            "text":        ["text_encoder.", "conditioner.embedders.", "clip.", "t5."],
            "text2":       ["text_encoder_2."],
            "unet":        ["transformer."],
            "clip":        ["text_encoder.", "text_encoder_2.", "conditioner.embedders.", "clip.", "t5."],
            "clip-l":      ["text_encoder."],
            "clip-g":      ["text_encoder_2."],
        }

    if arch.get("XL", False):
        return {
            "unet":   ["model.diffusion_model."],
            "vae":    ["first_stage_model."],
            "clip-l": [
                "conditioner.embedders.0.",
                "text_encoders.encoder_l.",
                "clip_l.", "clip_l/"
            ],
            "clip-g": [
                "conditioner.embedders.1.",
                "text_encoders.encoder_g.",
                "clip_g.", "clip_g/"
            ],
            "clip": [
                "conditioner.embedders.",
                "text_encoders.", "clip_l.", "clip_g."
            ],
            "text": [
                "conditioner.embedders.",
                "text_encoders."
            ],
            "text2": [],
            "transformer": ["model.diffusion_model."],
        }

    if arch.get("ZI", False):
        return {
            "unet":        ["model.diffusion_model."],
            "transformer": ["model.diffusion_model."],
            "vae":         ["first_stage_model.", "vae."],
            "text":        ["text_encoders.qwen3_4b.transformer.", "qwen3_4b.", "cap_embedder.", "text_encoders.qwen3_4b."],
            "text2":       [],
            "clip":        ["text_encoders.qwen3_4b.transformer.", "qwen3_4b.", "cap_embedder.", "model.diffusion_model.cap_embedder.", "text_encoders.qwen3_4b."],
            "clip-l":      ["qwen3_4b.", "text_encoders.qwen3_4b."],
            "clip-g":      ["cap_embedder.", "model.diffusion_model.cap_embedder."],
        }
        
    if arch.get("AM", False):
        return {
            "unet":        ["model.diffusion_model."],
            "transformer": ["model.diffusion_model."],
            "vae":         ["first_stage_model.", "vae."],
            "text":        ["text_encoders.qwen3_06b.", "text_encoders.qwen3_06b_base.", "cond_stage_model.qwen3_06b."],
            "text2":       [],
            "clip":        ["text_encoders.qwen3_06b.", "text_encoders.qwen3_06b_base.", "cond_stage_model.qwen3_06b."],
            "clip-l":      ["text_encoders.qwen3_06b.", "text_encoders.qwen3_06b_base.", "cond_stage_model.qwen3_06b."],
            "clip-g":      [],
        }

    # SD1.x / SD2.x (non-XL, non-Flux, non-ZI)
    return {
        "unet": ["model.diffusion_model."],
        "vae":  ["first_stage_model."],
        "clip": ["cond_stage_model.", "clip."],
        "text": ["cond_stage_model.", "clip."],
        "text2": [],
        "transformer": ["model.diffusion_model."],
    }


def _key_belongs_to_component(key: str, prefixes: list[str]) -> bool:
    for p in prefixes:
        if key.startswith(p):
            return True
        if p.endswith(".") and key.startswith(p[:-1] + "/"):
            return True
    return False

def _best_prefix_match(k: str, prefixes: list[str]) -> str | None:
    best = None
    best_len = -1
    for p in prefixes:
        if k.startswith(p) and len(p) > best_len:
            best = p
            best_len = len(p)
    return best

def _suffix_candidates(k: str):
    cands = [k]
    if k.startswith("transformer."):
        cands.append(k[len("transformer."):])
    if k.startswith("transformer.model."):
        cands.append(k[len("transformer."):])
    if not k.startswith("model."):
        cands.append("model." + k)
    seen = set()
    out = []
    for x in cands:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def normalize_external_text_encoder(theta_src: dict, arch: dict) -> dict:
    if not isinstance(theta_src, dict) or not theta_src:
        return theta_src

    if arch.get("AM", False):
        te_root = "text_encoders.qwen3_06b."
        raw_model_prefix = te_root + "transformer."
        ok_prefixes = (
            "text_encoders.qwen3_06b.",
            "text_encoders.qwen3_06b_base.",
            "cond_stage_model.qwen3_06b.",
            "qwen3_06b.",
            "qwen3_06b_base.",
        )
    elif arch.get("ZI", False):
        te_root = "text_encoders.qwen3_4b."
        raw_model_prefix = te_root + "transformer."
        ok_prefixes = (
            "text_encoders.qwen3_4b.",
            "cond_stage_model.qwen3_4b.",
            "qwen3_4b.",
        )
    else:
        return theta_src

    def is_raw_qwen_key(k: str) -> bool:
        return k.startswith((
            "model.layers.", "model.embed_tokens.", "model.norm.", "lm_head.",
            "transformer.model.layers.", "transformer.model.embed_tokens.", "transformer.model.norm.", "transformer.lm_head.",
        )) or (k in ("logit_scale",))

    def map_key(k: str) -> str:
        if k.startswith("cond_stage_model.qwen3_06b.") and arch.get("AM", False):
            return te_root + k[len("cond_stage_model.qwen3_06b."):]
        if k.startswith("cond_stage_model.qwen3_4b.") and arch.get("ZI", False):
            return te_root + k[len("cond_stage_model.qwen3_4b."):]
        if k.startswith("qwen3_06b.") and arch.get("AM", False):
            return te_root + k[len("qwen3_06b."):]
        if k.startswith("qwen3_4b.") and arch.get("ZI", False):
            return te_root + k[len("qwen3_4b."):]

        if k.startswith(ok_prefixes):
            return k

        if k == "logit_scale":
            return te_root + "logit_scale"

        if k.startswith("transformer."):
            return te_root + k

        if k.startswith("model.") or k.startswith("lm_head."):
            return raw_model_prefix + k

        return k

    has_raw = any(is_raw_qwen_key(k) for k in theta_src.keys())
    has_any_ok = any(k.startswith(ok_prefixes) for k in theta_src.keys())

    if not has_raw and has_any_ok:
        return theta_src

    out = {}
    for k, v in theta_src.items():
        nk = map_key(k)
        out[nk] = v
    return out

@torch.inference_mode()
def _swap_components_inplace(
    theta_dst: dict,
    theta_src: dict,
    components: set[str],
    arch: dict,
    *,
    src_only: set[str] | None = None,
):
    if all(not v for v in arch.values()):
        arch, theta_src = detect_arch(theta_src)

    pref = _component_prefix_map(arch)

    selected = {c for c in (components or set()) if c in pref and pref.get(c)}
    prefixes = [p for c in selected for p in pref.get(c, [])]

    if not prefixes:
        return 0, 0, 0, theta_dst

    only = set(src_only or set())
    do_only = bool(only & selected)

    suffix_map = None
    if do_only:
        p_sorted = sorted(prefixes, key=len, reverse=True)
        suffix_map = {}
        for kd in theta_dst.keys():
            mp = _best_prefix_match(kd, p_sorted)
            if mp is None:
                continue
            suf = kd[len(mp):]
            if suf and (suf not in suffix_map):
                suffix_map[suf] = kd

    moved, created, skipped_shape = 0, 0, 0

    for k, v in tqdm(theta_src.items(), desc="Swapping components...", unit="param"):
        if not do_only:
            if not _key_belongs_to_component(k, prefixes):
                continue
            k_use = k
        else:
            k_use = None
            if k in theta_dst:
                k_use = k
            else:
                for cand in _suffix_candidates(k):
                    kk = suffix_map.get(cand) if suffix_map is not None else None
                    if kk is not None:
                        k_use = kk
                        break
                    
                if k_use is None:
                    k_use = prefixes[0] + k

        if k_use in theta_dst:
            dv = theta_dst[k_use]
            if hasattr(dv, "shape") and hasattr(v, "shape") and tuple(dv.shape) != tuple(v.shape):
                skipped_shape += 1
                continue

            if isinstance(dv, torch.Tensor) and isinstance(v, torch.Tensor):
                vv = v
                if vv.dtype != dv.dtype or vv.device != dv.device:
                    vv = vv.to(device=dv.device, dtype=dv.dtype)
                dv.copy_(vv)
                theta_dst[k_use] = dv
            else:
                theta_dst[k_use] = v
            moved += 1
        else:
            theta_dst[k_use] = v
            created += 1

    return moved, created, skipped_shape, theta_dst


def _is_clip_key(key: str, arch: dict) -> bool:
    if arch.get("FLUX", False):
        prefixes = ["text_encoder.", "text_encoder_2.", "conditioner.embedders.", "clip.", "t5."]
    elif arch.get("XL", False):
        prefixes = ["conditioner.embedders.", "text_encoders.", "clip_l.", "clip_g."]
    elif arch.get("ZI", False):
        prefixes = ["qwen3_4b.", "cap_embedder.","text_encoders.qwen3_4b.","model.diffusion_model.cap_embedder."]
    elif arch.get("AM", False):
        prefixes = [
            "text_encoders.qwen3_06b.", "text_encoders.qwen3_06b_base.",
            "cond_stage_model.qwen3_06b.",
            "qwen3_06b.", "qwen3_06b_base.",
        ]
    else:
        prefixes = ["cond_stage_model.", "clip."]
    return any(key.startswith(p) for p in prefixes)


def _tensor_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a32 = a.detach().float().view(-1); b32 = b.detach().float().view(-1)
    if a32.numel() == 0:
        return torch.tensor(1.0, device=a.device)
    num = torch.dot(a32, b32)
    den = (a32.norm() * b32.norm()).clamp_min(1e-12)
    return (num / den).clamp(-1.0, 1.0)


def _maybe_skip_small_norm_bias_for_clipxor(key: str, tens: torch.Tensor) -> bool:
    if tens.numel() < 128:
        return True
    lk = key.lower()
    if lk.endswith(".bias") or ".bias" in lk:
        return True
    if ("norm" in lk) or (".ln" in lk) or (".bn" in lk):
        return True
    if ("emb" in lk) or ("pos" in lk):
        return True
    if ("text_projection" in lk) or ("logit_scale" in lk):
        return True
    return False


def _is_small_or_norm_or_bias(key, tens):
    n = tens.numel()
    if n < 128:
        return True
    k = key.lower()
    if (k.endswith(".bias") or ".bias" in k or "norm" in k or "ln" in k or "bn" in k):
        return True
    if "emb" in k or "pos" in k:
        return True
    return False


def _collect_clipxor_targets(theta_base: dict, theta_other: dict,
                             arch: dict) -> list[str]:
    targets = []
    for k, A in theta_base.items():
        if not _is_clip_key(k, arch):
            continue
        B = theta_other.get(k)
        if getattr(A, "shape", None) != getattr(B, "shape", None):
            continue
        if _maybe_skip_small_norm_bias_for_clipxor(k, A):
            continue
        targets.append(k)
    return targets

def _clip_roots_for_arch(arch: dict) -> list[str]:
    if arch.get("FLUX", False):
        # Flux: T5 / text encoders
        return [
            "text_encoder.", "text_encoder_2.", "conditioner.embedders.", "clip.", "t5."
        ]
    if arch.get("XL", False):
        # SDXL: CLIP-L / CLIP-G
        return [
            "conditioner.embedders.0.", "conditioner.embedders.1.",
            "text_encoders.encoder_l.", "text_encoders.encoder_g.",
            "clip_l.", "clip_g."
        ]
    if arch.get("ZI", False):
        # Z-Image: Qwen + caption embedder
        return [
            "qwen3_4b.",
            "cap_embedder.",
        ]
    if arch.get("AM", False):
        return [
            "text_encoders.qwen3_06b.", "text_encoders.qwen3_06b_base.",
            "cond_stage_model.qwen3_06b.",
            "qwen3_06b.", "qwen3_06b_base.",
        ]
    # SD1.x / SD2.x (non-XL)
    return ["cond_stage_model.", "clip."]


def _iter_clip_items(sd: dict, arch: dict):
    roots = _clip_roots_for_arch(arch)
    for k, v in sd.items():
        for r in roots:
            if k.startswith(r):
                # canonical suffix after the first root occurrence
                suffix = k[len(r):]
                yield (k, r, suffix, v)
                break

def _collect_clip_pairs_by_suffix(sd_a: dict, sd_b: dict,
                                  arch_a: dict, arch_b: dict):
    # map suffix -> (orig_key, tensor) for A and B separately
    map_a = {}
    for k, _, suf, v in _iter_clip_items(sd_a, arch_a):
        map_a[suf] = (k, v)
    map_b = {}
    for k, _, suf, v in _iter_clip_items(sd_b, arch_b):
        map_b[suf] = (k, v)

    # intersect by suffix and by matching shape
    pairs = []
    for suf, (ka, va) in map_a.items():
        kb_v = map_b.get(suf)
        if kb_v is None:
            continue
        kb, vb = kb_v
        if getattr(va, "shape", None) == getattr(vb, "shape", None):
            pairs.append((suf, ka, kb))  # use A's key for writing back
    return pairs  # list of (suffix, key_in_A, key_in_B)

def _clip_tier_for_xl(key: str) -> str:
    k = key.lower()
    if ("clip_l" in k) or ("text_model" in k and "clip" in k) or ("conditioner.embedders.0" in k):
        return "clip-l"
    if ("clip_g" in k) or ("text2_model" in k and "clip" in k) or ("open_clip" in k)  or ("conditioner.embedders.1" in k):
        return "clip-g"
    return "other"

def _clip_tier_for_flux(key: str) -> str:
    k = key.lower()
    if ("t5" in k) or ("text_encoder" in k) or ("textencoder" in k) or ("conditioner.embedders" in k):
        return "t5"
    if ("clip" in k) and ("text" in k or "emb" in k or "proj" in k):
        return "clip"
    return "other"

def _clip_tier_for_zi(key: str) -> str:
    k = key.lower()
    if ("qwen3_4b" in k):
        return "qwen3_4b"
    if ("cap_embedder" in k):
        return "cap_embedder"
    return "other"

def _norm_stats(t: torch.Tensor, eps: float = 1e-6):
    v = t.detach().float().reshape(-1)
    if v.numel() == 0:
        return v.new_tensor(0.0), v.new_tensor(1.0)

    std, mean = torch.std_mean(v, correction=0)  # PyTorch 2.x
    return mean, std.clamp_min(eps)

def _apply_stat_alignment(out: torch.Tensor, ref: torch.Tensor, strength: float = 0.5):
    m_ref, s_ref = _norm_stats(ref)
    m_out, s_out = _norm_stats(out)
    aligned = (out - m_out) / s_out * s_ref + m_ref
    return out * (1 - strength) + aligned * strength

def _topk_mask(delta: torch.Tensor, k_frac: float) -> torch.Tensor:
    if delta.numel() == 0:
        return torch.zeros_like(delta, dtype=torch.float32)
    k = max(int(delta.numel() * float(k_frac)), 1)
    flat = delta.detach().float().abs().view(-1)
    if k >= flat.numel():
        mask = torch.ones_like(flat)
    else:
        thresh = torch.kthvalue(flat, flat.numel() - k + 1).values
        mask = (flat >= thresh).float()
    return mask.view_as(delta)

def _clipxor_semi_hard_blend(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    hardness: float = 0.7,
    use_cosine_gate: bool = True,
    keep_stats: bool = True
) -> torch.Tensor:
    same_sign = torch.sign(A) == torch.sign(B)
    overlap = torch.where(same_sign, torch.sign(A) * torch.minimum(A.abs(), B.abs()), torch.zeros_like(A))
    U = A + B - overlap
    delta = (U - A)

    if delta.numel() == 0:
        return A

    k_frac = 0.10 + 0.45 * float(hardness)
    alpha  = 0.12 + 0.38 * float(hardness)
    tau    = 0.35 - 0.20 * float(hardness)
    sharp  = 0.12 + 0.08 * (1.0 - float(hardness))
    tr_budget = 0.12 + 0.38 * float(hardness)
    stat_strength = 0.65 - 0.30 * float(hardness)

    m = _topk_mask(delta, k_frac=k_frac)
    delta_sel = (delta.detach().float() * m).to(A.dtype)

    if use_cosine_gate:
        cos = _tensor_cosine(A, U)
        g = torch.sigmoid((cos - tau) / max(sharp, 1e-3)).to(A.dtype)
    else:
        g = A.new_tensor(1.0)

    out = A + (alpha * g) * delta_sel

    a_norm = A.detach().float().norm()
    d_norm = (out - A).detach().float().norm().clamp_min(1e-12)
    budget = tr_budget * a_norm
    if d_norm > budget:
        scale = (budget / d_norm).to(out.dtype)
        out = A + (out - A) * scale

    if keep_stats and stat_strength > 0.0:
        out = _apply_stat_alignment(out, A, strength=float(stat_strength))

    return out

def _hp_kernel_inplace(w: torch.Tensor, strength: float) -> torch.Tensor:
    if w.ndim != 4:
        return w
    kh, kw = w.shape[-2], w.shape[-1]
    if kh < 3 or kw < 3:
        return w
    x = w.float()
    C = x.shape[0] * x.shape[1]
    x2 = x.reshape(1, C, kh, kw)
    blur = F.avg_pool2d(x2, kernel_size=3, stride=1, padding=1)
    hp = x2 - blur
    x3 = x2 + (strength * hp)
    return x3.reshape_as(x).to(dtype=w.dtype)


def _detail_kernel_inplace(w: torch.Tensor, strength: float) -> torch.Tensor:
    if w.ndim != 4:
        return w
    kh, kw = w.shape[-2], w.shape[-1]
    if kh < 3 or kw < 3:
        return w

    x = w.float()
    C = x.shape[0] * x.shape[1]
    x2 = x.reshape(1, C, kh, kw)
    blur = F.avg_pool2d(x2, kernel_size=3, stride=1, padding=1)
    hp = x2 - blur

    if strength >= 0:
        y = x2 + strength * hp
    else:
        amt = min(abs(float(strength)), 1.0) * 0.22
        y = torch.lerp(x2, blur, amt)

    return y.reshape_as(x).to(dtype=w.dtype)

def _is_conv_weight(key: str, tens: torch.Tensor) -> bool:
    return tens.ndim == 4 and (key.endswith(".weight") or ".weight" in key)

def _is_bias(key: str, tens: torch.Tensor) -> bool:
    return tens.ndim == 1 and (key.endswith(".bias") or ".bias" in key)

def _is_weight_like(key: str, tens: torch.Tensor) -> bool:
    return tens.ndim >= 2 and (key.endswith(".weight") or ".weight" in key)

def _is_norm_key(key: str) -> bool:
    kl = key.lower()
    if not (key.endswith(".weight") or key.endswith(".bias") or ".weight" in key or ".bias" in key):
        return False
    return ("norm" in kl) or (".ln" in kl) or (".bn" in kl)

def _is_attn1_vout(key: str) -> bool:
    kl = key.lower()
    return (
        "attn1.to_v" in kl or
        "attn1.to_out" in kl or
        "self_attn.v_proj" in kl or
        "self_attn.out_proj" in kl
    )

def _is_resnetish(key: str) -> bool:
    return ("resnets" in key) or ("in_layers" in key) or ("out_layers" in key) or (".conv1." in key) or (".conv2." in key)

def _is_proj_ff(key: str) -> bool:
    return ("proj_in" in key) or ("proj_out" in key) or ("ff.net" in key)

def _is_unet_out(key: str) -> bool:
    kl = key.lower()
    return (
        ("model.diffusion_model.out.0." in kl) or
        ("model.diffusion_model.out.2." in kl) or
        # Anima / DiT-like final RGB head
        ("model.diffusion_model.final_layer." in kl) or
        kl.startswith("final_layer.") or
        kl.startswith("net.final_layer.")
    )

def _is_vae_rgb(key: str) -> bool:
    kl = key.lower()
    return (
        ("first_stage_model.decoder.conv_out." in kl) or
        ("vae.decoder.conv_out." in kl) or
        ("model.vae.decoder.conv_out." in kl) or
        ("model.first_stage_model.decoder.conv_out." in kl)
    )


def _am_block_band(blk: str) -> float:
    """Broad tone weighting for Anima L00-L27 blocks."""
    if not isinstance(blk, str) or not blk.startswith("L"):
        return 0.0
    try:
        n = int(blk[1:])
    except Exception:
        return 0.0
    if 20 <= n <= 27:
        return 1.20
    if 12 <= n <= 19:
        return 1.00
    if 8 <= n <= 11:
        return 0.75
    if 4 <= n <= 7:
        return 0.55
    return 0.35


def _am_saturation_finetune_tensor(key: str, tens: torch.Tensor, blk: str, sat: float):
    """
    Gentle diffusion-side tone correction for Anima.

    Real RGB saturation is handled by VAE conv_out via
    apply_vae_saturation_inplace(). This only damp/boosts AM color/tone
    sensitive transformer tensors so NoIn/merge fine passes have AM coverage.
    """
    if not isinstance(tens, torch.Tensor) or (not tens.is_floating_point()):
        return tens

    band = _am_block_band(blk)
    if band <= 0:
        return tens

    kl = key.lower()
    down = max(0.0, 1.0 - float(sat))
    up = max(0.0, float(sat) - 1.0)
    if down == 0.0 and up == 0.0:
        return tens

    coeff = 0.0
    if ("mlp.layer1" in kl) or ("mlp.layer2" in kl) or ("adaln_modulation_mlp" in kl):
        coeff = 0.10
    elif (("v_proj" in kl) or ("output_proj" in kl)) and (("self_attn" in kl) or ("cross_attn" in kl)):
        coeff = 0.07
    elif ("q_norm" in kl) or ("k_norm" in kl) or ("t_embedding_norm" in kl):
        coeff = 0.04
    elif ("adaln_modulation_self_attn" in kl) or ("adaln_modulation_cross_attn" in kl):
        coeff = 0.035

    if coeff <= 0.0:
        return tens

    scale = 1.0 - (down * coeff * band) + (up * coeff * band)
    scale = max(0.70, min(1.30, scale))
    return (tens.float() * scale).to(dtype=tens.dtype)


def _finetune_inplace(key, tens, fine, arch: dict):
    if fine == "" or fine is None:
        return tens
    
    if isinstance(fine, (list, tuple)) and len(fine) >= 8:
        fine = {
            "mode": "image_tone_v2",
            "noise1": float(fine[0]),
            "noise2": float(fine[1]),
            "noise3": float(fine[2]),
            "contrast": float(fine[3]),
            "brightness": float(fine[4]),
            "r": float(fine[5]),
            "g": float(fine[6]),
            "b": float(fine[7]),
            "saturation": float(fine[8]) if len(fine) > 8 else 1.0,
        }

    if isinstance(fine, dict):
        if fine.get("mode") == "image_tone_v2":
            dn1 = float(fine.get("noise1", 0.0))
            dn2 = float(fine.get("noise2", 0.0))
            dn3 = float(fine.get("noise3", 0.0))
            ct  = float(fine.get("contrast", 0.0))
            br  = float(fine.get("brightness", 0.0))
            rr  = float(fine.get("r", 0.0))
            gg  = float(fine.get("g", 0.0))
            bb  = float(fine.get("b", 0.0))
            sat = float(fine.get("saturation", 1.0))

            # detail / clarity band controls
            dn1_k = dn1 * 0.06
            dn2_k = dn2 * 0.05
            dn3_k = dn3 * 0.04

            # signed tone controls
            soft = max(-ct, 0.0)   # contrast down / black lift up
            hard = max(ct, 0.0)    # contrast up / black deepen

            ct_scale_out   = 1.0 - soft * 0.018 + hard * 0.03
            ct_scale_out08 = 1.0 - soft * 0.015 + hard * 0.025
            bright_add     = br * 0.008
            black_lift     = soft * (4.0 / 255.0) - hard * (3.0 / 255.0)
            gamma_soft     = 1.0 + soft * 0.018 - hard * 0.02

            r_k = 1.0 + rr * 0.05
            g_k = 1.0 + gg * 0.05
            b_k = 1.0 + bb * 0.05

            left, right = blockfromkey(key, arch)
            blk = right

            # --- VAE RGB / small brightness carry / direct saturation ---
            if _is_vae_rgb(key):
                if _is_conv_weight(key, tens) and tens.shape[0] >= 3:
                    w = tens.float()
                    if abs(sat - 1.0) >= 1e-6:
                        A = _rgb_sat_matrix(sat, w.device, torch.float32)
                        w[:3] = torch.einsum("ij,jchw->ichw", A, w[:3])
                    w[0] *= r_k
                    w[1] *= g_k
                    w[2] *= b_k
                    # w *= (1.0 + ct * 0.01)
                    return w.to(dtype=tens.dtype)

                if _is_bias(key, tens) and tens.shape[0] >= 3:
                    b = tens.float()
                    if abs(sat - 1.0) >= 1e-6:
                        A = _rgb_sat_matrix(sat, b.device, torch.float32)
                        b[:3] = torch.einsum("ij,j->i", A, b[:3])
                    base_add = bright_add + black_lift
                    b[0] = b[0] * r_k + base_add
                    b[1] = b[1] * g_k + base_add
                    b[2] = b[2] * b_k + base_add
                    return b.to(dtype=tens.dtype)

                return tens

            # --- Anima diffusion-side saturation/tone support ---
            if arch.get("AM", False) and abs(sat - 1.0) >= 1e-6:
                tuned = _am_saturation_finetune_tensor(key, tens, blk, sat)
                if tuned is not tens:
                    return tuned

            # --- UNet final out: primary global contrast / black point ---
            if _is_unet_out(key):
                if _is_weight_like(key, tens):
                    scale = ct_scale_out * gamma_soft
                    return (tens.float() * scale).to(dtype=tens.dtype)
                if _is_bias(key, tens):
                    # add = bright_add + black_lift + shadow_lift * 0.5
                    # return (tens.float() + add).to(dtype=tens.dtype)
                    return tens

            # --- tone shaping on norm blocks ---
            if _is_norm_key(key) and blk in ("IN00", "OUT00", "OUT01", "OUT02", "OUT08"):
                if key.endswith(".weight") or ".weight" in key:
                    scale = 1.0 - soft * 0.015 + hard * 0.02
                    if blk == "OUT08":
                        scale += (-soft * 0.01 + hard * 0.01)
                    return (tens.float() * scale).to(dtype=tens.dtype)

                if key.endswith(".bias") or ".bias" in key:
                    # add = bright_add * 0.35
                    # add += shadow_lift * (0.35 if blk in ("OUT00", "OUT01", "OUT02") else 0.2)
                    # if blk == "OUT08":
                    #     add += black_lift * 0.5
                    # elif blk == "IN00":
                    #     add += black_lift * 0.2
                    # return (tens.float() + add).to(dtype=tens.dtype)
                    return tens

            # --- OUT08 proj/ff: gamma / black rolloff ---
            if blk == "OUT08" and _is_proj_ff(key) and _is_weight_like(key, tens):
                scale = ct_scale_out08
                return (tens.float() * scale).to(dtype=tens.dtype)

            # --- OUT07/OUT08 attn1 v/out: lighting / perceived contrast ---
            if blk == "OUT07" and hard > 0 and _is_attn1_vout(key) and _is_weight_like(key, tens):
                scale = 1.0 + hard * 0.01
                return (tens.float() * scale).to(dtype=tens.dtype)

            # --- detail / local contrast / clarity ---
            if not _is_conv_weight(key, tens):
                return tens

            if not _is_resnetish(key):
                return tens

            strength = 0.0
            # NOISE1: OUT07-OUT08
            if blk == "OUT08":
                strength = dn1_k
            # NOISE2: OUT05-OUT07
            elif blk in ("OUT05", "OUT06"):
                strength = dn2_k
            # NOISE3: OUT03-OUT05 + IN00
            elif blk in ("OUT03", "OUT04", "IN00"):
                strength = dn3_k

            if strength != 0.0:
                return _detail_kernel_inplace(tens, strength)

            return tens

        mul = fine.get("mul", {}) or {}
        m = 1.0

        if any(s in key for s in ("double_block", "double_blocks", "db.")):
            m *= float(mul.get("double_block", 1.0))
        if any(s in key for s in (".img_in", "image_in", "image_proj")):
            m *= float(mul.get("img_in", 1.0))
        if any(s in key for s in (".txt_in", "context_in", "text_in", "clip_proj")):
            m *= float(mul.get("txt_in", 1.0))
        if any(s in key for s in ("time_in", "time_embed", "timestep", "vector_in")):
            m *= float(mul.get("time", 1.0))
        if any(s in key for s in (".out.", "final_layer", "vector_out")) or key.endswith(".out"):
            m *= float(mul.get("out", 1.0))

        if m != 1.0:
            tens = tens * torch.as_tensor(m, device=tens.device, dtype=tens.dtype)

        add = float(fine.get("add", 0.0) or 0.0)
        if add and (key.endswith(".bias") or ".bias" in key):
            tens = tens + torch.as_tensor(add, device=tens.device, dtype=tens.dtype)

        return tens

    # ---- list mode ----
    if isinstance(fine, list):
        # ===== NEW: 8 sliders for SDXL/SD15 =====
        if len(fine) >= 8:
            dn1, dn2, dn3, ct, br, rr, gg, bb = [float(x) for x in fine[:8]]
            sat = float(fine[8]) if len(fine) > 8 else 1.0
            
            dn1_k = dn1 * 0.12
            dn2_k = dn2 * 0.10
            dn3_k = dn3 * 0.08
            ct_k  = 1.0 + ct * 0.06
            br_k  = br * 0.02
            r_k, g_k, b_k = (1.0 + rr * 0.05, 1.0 + gg * 0.05, 1.0 + bb * 0.05)

            # --- VAE: Brightness / RGB / direct saturation ---
            if _is_vae_rgb(key):
                if _is_conv_weight(key, tens) and tens.shape[0] >= 3:
                    w = tens.float()
                    if abs(sat - 1.0) >= 1e-6:
                        A = _rgb_sat_matrix(sat, w.device, torch.float32)
                        w[:3] = torch.einsum("ij,jchw->ichw", A, w[:3])
                    w[0] *= r_k
                    w[1] *= g_k
                    w[2] *= b_k
                    return w.to(dtype=tens.dtype)
                if _is_bias(key, tens) and tens.shape[0] >= 3:
                    b = tens.float()
                    if abs(sat - 1.0) >= 1e-6:
                        A = _rgb_sat_matrix(sat, b.device, torch.float32)
                        b[:3] = torch.einsum("ij,j->i", A, b[:3])
                    b[0] = b[0] * r_k + br_k
                    b[1] = b[1] * g_k + br_k
                    b[2] = b[2] * b_k + br_k
                    return b.to(dtype=tens.dtype)
                return tens

            # --- Anima diffusion-side saturation/tone support ---
            left, right = blockfromkey(key, arch)
            blk = right
            if arch.get("AM", False) and abs(sat - 1.0) >= 1e-6:
                tuned = _am_saturation_finetune_tensor(key, tens, blk, sat)
                if tuned is not tens:
                    return tuned

            # --- UNet Contrast / Brightness ---
            if _is_unet_out(key):
                if _is_conv_weight(key, tens):
                    return (tens.float() * ct_k).to(dtype=tens.dtype)
                if _is_bias(key, tens):
                    return (tens.float() + br_k).to(dtype=tens.dtype)

            # --- Detail/Noise ---
            left, right = blockfromkey(key, arch)
            blk = right

            if not _is_conv_weight(key, tens):
                return tens

            if not (_is_resnetish(key) or _is_proj_ff(key)):
                return tens

            strength = 0.0
            # DN1: OUT07-OUT08
            if blk in ("OUT07", "OUT08"):
                strength = dn1_k
            # DN2: OUT05-OUT07
            elif blk in ("OUT05", "OUT06", "OUT07"):
                strength = dn2_k
            # DN3: OUT03-OUT05 + IN00
            elif blk in ("OUT03", "OUT04", "OUT05", "IN00"):
                strength = dn3_k

            if strength != 0.0:
                return _hp_kernel_inplace(tens, strength)

            return tens

        # ===== OLD: keep existing behavior (FINETUNES 6 keys) =====
        idx = next((i for i, pat in enumerate(FINETUNES) if pat in key), -1)
        if idx == -1:
            return tens

        if idx < 5:
            return tens * torch.as_tensor(fine[idx], device=tens.device, dtype=tens.dtype)
        else:
            try:
                add = torch.as_tensor(fine[5], device=tens.device, dtype=tens.dtype)
                return tens + add
            except Exception:
                if isinstance(fine[5], (list, tuple)) and len(fine[5]) > 0:
                    add = torch.as_tensor(fine[5][0], device=tens.device, dtype=tens.dtype)
                    return tens + add
                else:
                    add = torch.as_tensor(float(fine[5]) if fine[5] is not None else 0.0,
                                          device=tens.device, dtype=tens.dtype)
                    return tens + add

    return tens

def trim_delta(delta: torch.Tensor, percentile: float = 0.5) -> torch.Tensor:
    if delta.dim() == 4 and min(delta.shape[-2:]) > 2:
        blurred = F.avg_pool2d(delta, kernel_size=3, stride=1, padding=1)
        delta = delta * (1 - percentile) + blurred * percentile
    return delta

def prune_extras_vs_model1(theta_base, theta_ref):
    to_delete = []
    for k in theta_base.keys():
        if ("model" not in k) and ("text_encoders" not in k):
            continue
        if k not in theta_ref:
            to_delete.append(k)

    for k in to_delete:
        del theta_base[k]

    if to_delete:
        print(f"[TF] Pruned {len(to_delete)} keys that were not present in model_1")

    return theta_base

class PermutationSpec(NamedTuple):
    perm_to_axes: dict   # {perm_name: [(weight_key, axis), ...]}
    axes_to_perm: dict   # {weight_key: (perm_for_axis0, perm_for_axis1, ...)}


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def unet_permutation_spec(isxl: bool) -> PermutationSpec:
    def conv(name: str, p_in, p_out):
        # weight: (out, in), bias: (out,)
        return {
            f"{name}.weight": (p_out, p_in),
            f"{name}.bias":   (p_out,),
        }

    def norm(name: str, p):
        # weight/bias: (p,)
        return {
            f"{name}.weight": (p,),
            f"{name}.bias":   (p,),
        }

    def dense(name: str, p_in, p_out, bias: bool = True):
        d = {f"{name}.weight": (p_out, p_in)}
        if bias:
            d[f"{name}.bias"] = (p_out,)
        return d

    def easyblock(name: str, p_in, p_out):
        p_inner  = f"P_{name}_inner"
        p_inner2 = f"P_{name}_inner2"
        p_inner3 = f"P_{name}_inner3"
        p_inner4 = f"P_{name}_inner4"
        return {
            **norm(f"{name}.in_layers.0", p_in),
            **conv(f"{name}.in_layers.2", p_in, p_inner),
            **dense(f"{name}.emb_layers.1", p_inner2, p_inner3, bias=True),
            **norm(f"{name}.out_layers.0", p_inner4),
            **conv(f"{name}.out_layers.3", p_inner4, p_out),
        }

    filename = "sdxl_perm.json" if isxl else "sd_perm.json"
    with open(base_path(filename), "r", encoding="utf-8") as f:
        spec = json.load(f)

    axes_to_perm = {}
    def bg(idx: int):
        return None if idx < 0 else f"P_bg{idx}"

    for key, value in spec.items():
        if "skip" in value:
            axes_to_perm[key] = (None, None, None, None)
            continue

        if "conv" in value:
            i = int(value["conv"])
            axes_to_perm.update(conv(key, bg(i), f"P_bg{i + 1}"))
        elif "norm" in value:
            i = int(value["norm"])
            axes_to_perm.update(norm(key, bg(i)))
        elif "dense" in value:
            i = int(value["dense"])
            axes_to_perm.update(
                dense(key, bg(i), f"P_bg{i + 1}", bool(value.get("bias", True)))
            )
        elif "eb" in value:
            i = int(value["eb"])
            axes_to_perm.update(easyblock(key, bg(i), f"P_bg{i + 1}"))
            
    return permutation_spec_from_axes_to_perm(axes_to_perm)

def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    w = params[k]

    for axis, p in enumerate(ps.axes_to_perm.get(k, [])):
        if axis == except_axis:
            continue

        if not p:
            continue
        
        if p not in perm:
            idx = torch.arange(w.shape[axis], device=w.device)
            perm[p] = idx
        else:
            idx = perm[p].to(w.device).long()

        axis_dim = w.shape[axis]

        if (
            idx.numel() != axis_dim
            or idx.min().item() < 0
            or idx.max().item() >= axis_dim
        ):
            idx = torch.arange(axis_dim, device=w.device)
            perm[p] = idx

        w = torch.index_select(w, axis, idx)

    return w



def apply_permutation(ps: PermutationSpec, perm: dict, params: dict) -> dict:
    return {k: get_permuted_param(ps, perm, k, params) for k in params}


def inner_matching(
    n: int,
    ps: PermutationSpec,
    p: str,
    params_a: dict,
    params_b: dict,
    usefp16: bool,
    progress: bool,
    number: int,
    linear_sum: float,
    perm: dict,
    device,
):
    dtype = torch.float16 if usefp16 else torch.float32
    A = torch.zeros((n, n), dtype=dtype, device=device)

    for wk, axis in ps.perm_to_axes.get(p, []):
        if wk not in params_a or wk not in params_b:
            continue

        w_a = params_a[wk]
        w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)

        w_a = torch.moveaxis(w_a, axis, 0).reshape(n, -1).to(device)
        w_b = torch.moveaxis(w_b, axis, 0).reshape(n, -1).T.to(device)

        if usefp16:
            w_a = w_a.half()
            w_b = w_b.half()

        try:
            A = A + (w_a @ w_b)
        except RuntimeError:
            A = A + (torch.dequantize(w_a) @ torch.dequantize(w_b))

    # Hungarian
    A_cpu = A.detach().cpu()
    ri, ci = linear_sum_assignment(A_cpu.numpy(), maximize=True)
    ri = torch.as_tensor(ri)
    ci = torch.as_tensor(ci)

    assert torch.equal(ri, torch.arange(len(ri))), "Unexpected row indices"

    eye = torch.eye(n, device=device)
    A_flat = A.flatten().float()

    oldL = torch.vdot(A_flat, eye[perm[p].long()].flatten())
    newL = torch.vdot(A_flat, eye[ci.long(), :].flatten())

    if usefp16:
        oldL = oldL.half()
        newL = newL.half()

    if (newL - oldL) != 0:
        linear_sum += float(abs(newL - oldL))
        number += 1

    improved = bool(newL > oldL + 1e-12)
    progress = progress or improved

    perm[p] = ci.to(device).float()

    return linear_sum, number, perm, progress

def weight_matching(
    ps: PermutationSpec,
    params_a: dict,
    params_b: dict,
    max_iter: int = 1,
    init_perm: dict | None = None,
    usefp16: bool = False,
    device: str | torch.device = "cpu",
    groups: list[str] | None = None,
):
    perm_sizes = {
        p: params_a[axes[0][0]].shape[axes[0][1]]
        for p, axes in ps.perm_to_axes.items()
        if axes and axes[0][0] in params_a
    }

    if init_perm is None:
        perm = {p: torch.arange(n, device=device) for p, n in perm_sizes.items()}
    else:
        perm = {p: v.to(device).long() for p, v in init_perm.items()}
    for p, axes in ps.perm_to_axes.items():
        if not axes:
            continue

        size = None
        for tensor_name, axis in axes:
            t = params_a.get(tensor_name)
            if t is None:
                t = params_b.get(tensor_name)
            if t is not None:
                size = t.shape[axis]
                break

        if size is not None:
            perm_sizes[p] = size

    if init_perm is None:
        perm = {
            p: torch.arange(n, device=device).float()
            for p, n in perm_sizes.items()
        }
    else:
        perm = {p: v.to(device).float() for p, v in init_perm.items()}

    special_layers = ["P_bg324"]
    target_groups = groups if groups is not None else special_layers
    target_groups = [g for g in target_groups if g in perm_sizes]

    linear_sum: float = 0.0
    number: int = 0

    if not target_groups or max_iter <= 0:
        return perm, 0.0

    for _ in tqdm(range(max_iter), desc="Weight matching"):
        random.shuffle(target_groups)
        progress = False

        for p in target_groups:
            n = perm_sizes[p]
            linear_sum, number, perm, progress = inner_matching(
                n,
                ps,
                p,
                params_a,
                params_b,
                usefp16,
                progress,
                number,
                linear_sum,
                perm,
                device,
            )

        if not progress:
            break

    average = float(linear_sum) / float(number) if number > 0 else 0.0
    return perm, average


def _filter_state_dict_by_components(theta: dict, components: set[str], arch: dict):
    pref = _component_prefix_map(arch)
    prefixes = [p for c in components for p in pref.get(c, []) if p]

    total = len(theta)
    if not prefixes:
        return {}, 0, total

    kept = {k: v for k, v in theta.items() if _key_belongs_to_component(k, prefixes)}
    return kept, len(kept), total


def _turbo_force_copy_key(k: str) -> bool:
    kl = k.lower()
    return (
        "alphas_cumprod" in kl or "betas" in kl or "posterior_" in kl or
        "sqrt_" in kl or "log_" in kl or "sigmas" in kl or
        "model_ema." in kl or "num_batches_tracked" in kl
    )

@torch.inference_mode()
def _scalar_ls_scale(tb: torch.Tensor, tc: torch.Tensor, sample: int = 2048) -> float:
    b = tb.detach().to(torch.float32).reshape(-1)
    c = tc.detach().to(torch.float32).reshape(-1)
    n = b.numel()
    if n == 0:
        return 1.0
    if n > sample:
        if sample <= 1:
            idx = torch.zeros((1,), device=b.device, dtype=torch.int64)
        else:
            idx = torch.arange(sample, device=b.device, dtype=torch.int64)
            idx = (idx * (n - 1)) // (sample - 1)
        idx.clamp_(0, n - 1)
        b = b.index_select(0, idx)
        c = c.index_select(0, idx)
    denom = float(torch.dot(c, c).clamp_min(1e-12))
    num   = float(torch.dot(b, c))
    s = num / denom
    if not np.isfinite(s):
        s = 1.0
    return s

def _is_cond_sensitive_key(k: str) -> bool:
    kl = k.lower()
    return (
        "attn2" in kl or "cross_attn" in kl or "crossattn" in kl or
        ".to_q" in kl or ".to_k" in kl or ".to_v" in kl or ".to_out" in kl or
        "q_proj" in kl or "k_proj" in kl or "v_proj" in kl or "out_proj" in kl or
        "encoder_hid_proj" in kl or
        "conditioner." in kl or "cond_stage_model." in kl or "text_encoders" in kl or
        "context_refiner" in kl or "cap_embedder" in kl
    )

@torch.inference_mode()
def turbo_convert_inplace(
    A: dict, B: dict, C: dict, resolver,
    *, deturbo: bool,
    vae_key: str,
    bake_vae_enabled: bool,
    fine=None,
):
    def allow_vae(k: str) -> bool:
        return bake_vae_enabled or (vae_key not in k)

    keys = set(A.keys()) | (set(B.keys()) & set(C.keys()))

    for k in tqdm(keys, desc=("DeTurbo" if deturbo else "Turbo") + " converting..."):
        tb = B.get(k, None)
        tc = C.get(k, None)

        if not (isinstance(tb, torch.Tensor) and isinstance(tc, torch.Tensor)):
            continue
        if tb.shape != tc.shape:
            continue

        if not allow_vae(k):
            continue

        ent = resolver(k)
        if ent is None:
            continue
        _, cur_a, _ = ent
        a = float(cur_a)
        if a == 0.0:
            continue

        ta = A.get(k, None)

        if _turbo_force_copy_key(k) or (not tb.is_floating_point()) or (not tc.is_floating_point()):
            tgt = tc if deturbo else tb
            if isinstance(ta, torch.Tensor) and ta.shape == tgt.shape and ta.is_floating_point() and tgt.is_floating_point():
                out = torch.lerp(ta.to(torch.float32), tgt.to(torch.float32), a).to(ta.dtype)
            else:
                out = tgt
            A[k] = _finetune_inplace(k, out, fine) if fine else out
            continue

        s = _scalar_ls_scale(tb, tc)
        hi = 6.0 if deturbo and _is_cond_sensitive_key(k) else 4.0
        s = float(np.clip(s, 0.25, hi))
        invs = 1.0 / max(s, 1e-6)

        if isinstance(ta, torch.Tensor) and ta.is_floating_point() and ta.shape == tb.shape:
            A32 = ta.detach().to(torch.float32)
        else:
            A32 = (tc.detach().to(torch.float32) if deturbo else tb.detach().to(torch.float32))

        B32 = tb.detach().to(torch.float32)
        C32 = tc.detach().to(torch.float32)

        if deturbo:
            # A' = A*(1-a + a*(1/s)) + a*(C - B/s)
            out32 = A32.mul(1.0 - a + a * invs).add_(C32 - B32 * invs, alpha=a)
        else:
            # A' = A*(1-a + a*s) + a*(B - s*C)
            out32 = A32.mul(1.0 - a + a * s).add_(B32 - C32 * s, alpha=a)
            
        def _soft_match_mean_std(out32: torch.Tensor, ref32: torch.Tensor, strength: float = 0.5, eps: float = 1e-6):
            varO, meanO = torch.var_mean(out32, unbiased=False)
            varR, meanR = torch.var_mean(ref32, unbiased=False)
            stdO = varO.sqrt().clamp_min(eps)
            stdR = varR.sqrt().clamp_min(eps)
            normed = (out32 - meanO) / stdO
            matched = normed * stdR + meanR
            return torch.lerp(out32, matched, float(strength))
        
        if deturbo and _is_cond_sensitive_key(k) and (not _is_small_or_norm_or_bias(k, tb)):
            COND_DAMP = 0.35 
            out32 = C32 + (out32 - C32) * COND_DAMP

            out32 = _soft_match_mean_std(out32, C32, strength=0.5)

        out = out32.to(ta.dtype if isinstance(ta, torch.Tensor) and ta.is_floating_point() else tb.dtype)
        A[k] = _finetune_inplace(k, out, fine) if fine else out

    fill = C if deturbo else B
    for k, v in fill.items():
        if k in A:
            continue
        if not allow_vae(k):
            continue
        A[k] = v

    return A

def _rgb_sat_matrix(s: float, device, dtype):
    s = float(s)
    m = (1.0 - s) / 3.0
    A = torch.tensor([
        [s + m,     m,     m],
        [    m, s + m,     m],
        [    m,     m, s + m],
    ], device=device, dtype=dtype)
    return A

@torch.inference_mode()
def apply_vae_saturation_inplace(sd: dict, vae_key: str, sat: float):
    if abs(float(sat) - 1.0) < 1e-12:
        return sd

    weight_suffixes = (
        "decoder.conv_out.weight",
        "decoder.conv_out.0.weight",
    )
    bias_suffixes = (
        "decoder.conv_out.bias",
        "decoder.conv_out.0.bias",
    )

    w_keys = [k for k in sd.keys() if any(k.endswith(suf) for suf in weight_suffixes)]
    b_keys = [k for k in sd.keys() if any(k.endswith(suf) for suf in bias_suffixes)]

    nW = nB = 0
    for k in w_keys:
        W = sd.get(k, None)
        if not isinstance(W, torch.Tensor) or (not W.is_floating_point()):
            continue
        if W.dim() != 4 or W.shape[0] != 3:  # out_ch must be RGB=3
            continue
        A = _rgb_sat_matrix(sat, W.device, torch.float32)
        W32 = W.detach().to(torch.float32)
        # W' = A @ W   (mix output RGB channels)
        W2 = torch.einsum("ij,jchw->ichw", A, W32)
        sd[k] = W2.to(W.dtype)
        nW += 1

    for k in b_keys:
        b = sd.get(k, None)
        if not isinstance(b, torch.Tensor) or (not b.is_floating_point()):
            continue
        if b.dim() != 1 or b.shape[0] != 3:
            continue
        A = _rgb_sat_matrix(sat, b.device, torch.float32)
        b32 = b.detach().to(torch.float32)
        b2 = torch.einsum("ij,j->i", A, b32)
        sd[k] = b2.to(b.dtype)
        nB += 1

    print(f"[vae_sat] applied sat={sat} to VAE conv_out: weights={nW}, bias={nB}")
    return sd

def _model_path(root: str, name: str | None) -> str | None:
    return None if name is None else normalize_path(os.path.join(root, name))


def _model_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _load_umodel(path: str, *, name: str | None = None, device: str = "cpu", model_type: str = "checkpoint", verify_hash: bool = True, cache_path: str | None = None) -> UnifiedModel:
    from model import UnifiedModel
    return UnifiedModel.from_file(path, name=name, device=device, model_type=model_type, verify_hash=verify_hash, cache_path=cache_path)


def _clone_info(src: UnifiedModel | None, *, name: str | None = None, path: str | None = None) -> ModelInfo:
    from model import ModelInfo
    info = src.info.clone() if src is not None else ModelInfo(model_type="checkpoint")
    if name is not None:
        info.name = name
    if path is not None:
        info.path = path
    return info


def _build_output_path(args) -> tuple[str, str, str]:
    fmt = "safetensors" if args.save_safetensors else "ckpt"
    output_name = args.output
    output_file = f"{output_name}.{fmt}"
    output_path = normalize_path(os.path.join(args.model_path, output_file))

    if os.path.isfile(output_path):
        if args.force:
            print(f"[force] Overwriting existing file: {output_path}")
            try:
                os.remove(output_path)
            except Exception as e:
                print(f"[force] Failed to remove existing file: {e}")
        else:
            i = 0
            while os.path.isfile(output_path):
                output_name = f"{args.output}_{i:02}"
                output_file = f"{output_name}.{fmt}"
                output_path = normalize_path(os.path.join(args.model_path, output_file))
                i += 1
            print(f"Assigned result checkpoint name as {output_file}\n")

    return output_name, output_file, output_path


def _save_umodel(model: UnifiedModel, path: str, *, args, metadata: dict | None = None) -> None:
    if metadata:
        model.info.metadata.update(metadata)
    model.save(
        path,
        no_metadata=bool(args.no_metadata),
        save_half=bool(args.save_half),
        save_quarter=bool(args.save_quarter),
        save_bhalf=bool(args.save_bhalf),
        prune=bool(args.prune),
        args=args,
    )


def _register_merge_parents(dst: UnifiedModel, *parents: UnifiedModel | None) -> None:
    for i, p in enumerate(parents):
        if p is not None:
            dst.register_parent(p, role=f"model_{i}")