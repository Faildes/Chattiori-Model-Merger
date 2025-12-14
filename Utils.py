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
import concurrent.futures as cf
from typing import List, Tuple, NamedTuple
from pathlib import Path
import torch.nn.functional as F
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

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
BLOCKIDXL = ["BASE"] + [f"IN{i}" for i in range(9)] + ["M"] + [f"OUT{i}" for i in range(9)] + ["VAE"]
BLOCKIDFLUX = ["CLIP", "T5", "IN"] + ["D{:002}".format(x) for x in range(19)] + ["S{:002}".format(x) for x in range(38)] + ["OUT"] # Len: 61
BLOCKIDZI = ["BASE","CONT","NOISE"] + [f"L{i:02}" for i in range(30)] + ["VAE"]

_re_inp = re.compile(r'\.input_blocks\.(\d+)\.')
_re_mid = re.compile(r'\.middle_block\.(\d+)\.')
_re_out = re.compile(r'\.output_blocks\.(\d+)\.')

FINETUNEX = ["IN", "OUT", "OUT2", "CONT", "BRI", "COL1", "COL2", "COL3"]
COLS = [[-1, 1/3, 2/3], [1, 1, 0], [0, -1, -1], [1, 0, 1]]
COLSXL = [[0, 0, 1], [1, 0, 0], [-1, -1, 0], [-1, 1, 0]]

PREFIXFIX = ("double_blocks","single_blocks","time_in","vector_in","txt_in")
PREFIX_M = "model.diffusion_model."
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


# Convenient ready-to-use constants
LBLOCKS_SDXL = make_lblocks_sdxl()
LBLOCKS_FLUX = make_lblocks_flux()
LBLOCKS_ZI   = make_lblocks_zi()

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

checkpoint_dict_skip_on_merge = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]
vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}

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

def base_path(path: str) -> str:
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

def fineman(fine, isxl=False, isflux=False):
    if isflux:
        mul = {
            "double_block": 1.0 + (fine[0] * 0.01) if len(fine) > 0 else 1.0,
            "img_in":       1.0 + (fine[1] * 0.01) if len(fine) > 1 else 1.0,
            "txt_in":       1.0 + (fine[2] * 0.01) if len(fine) > 2 else 1.0,
            "time":         1.0 + (fine[3] * 0.01) if len(fine) > 3 else 1.0,
            "out":          1.0 + (fine[4] * 0.01) if len(fine) > 4 else 1.0,
        }
        add = (fine[5] * 0.02) if len(fine) > 5 else 0.0
        return {"mul": mul, "add": add}
    r = [
        1 - fine[0] * 0.01,
        1 + fine[0] * 0.02,
        1 - fine[1] * 0.01,
        1 + fine[1] * 0.02,
        1 - fine[2] * 0.01,
        [fine[3] * 0.02] + colorcalc(fine[4:8], isxl)
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
        ratio, weights, info = ratios, [ratios]*25, f"{round(ratios,3)}"
    return weights, ratio, info


DTYPES = {torch.float32, torch.float64, torch.bfloat16}

def to_half(tensor, enable):
    return tensor.half() if enable and getattr(tensor, "dtype", None) in DTYPES else tensor

def to_half_k(sd, enable, benable, vae=None):
    if (enable or benable):
        for d in tqdm(list(sd.items()), desc="Half tensoring..."):
            k, v = d
            if ("model" in k or  "text_encoders" in k) and getattr(v, "dtype", None) in DTYPES:
                if vae:
                    if k.startswith(vae):
                        continue
                sd[k] = v.bfloat16() if benable else v.half()
    return sd


def upcast_fp8_state_dict(theta: dict, target_dtype: torch.dtype = torch.float16):
    if not FP8_DTYPES:
        return theta

    items = list(theta.items())

    fp8_items = [(k, v) for k, v in items
                 if isinstance(v, torch.Tensor) and v.dtype in FP8_DTYPES]
    if not fp8_items:
        return theta

    for k, v in tqdm(fp8_items,
                     total=len(fp8_items),
                     desc=f"Upcasting {len(fp8_items)} fp8 tensors..."):
        theta[k] = v.to(target_dtype)

    return theta


def to_quarter_k(theta: dict, enable: bool, prefer: str = "e4m3", vae=None):
    if not enable:
        return theta

    target_dtype = None
    if prefer == "e5m2" and FP8_E5M2 is not None:
        target_dtype = FP8_E5M2
    elif prefer == "e4m3" and FP8_E4M3 is not None:
        target_dtype = FP8_E4M3
    elif FP8_E4M3 is not None:
        target_dtype = FP8_E4M3
    elif FP8_E5M2 is not None:
        target_dtype = FP8_E5M2

    if target_dtype is None:
        print("[fp8] This PyTorch build has no float8 support; falling back to fp16/fp32.")
        return theta

    for k, v in theta.items():
        if ("model" in k or "text_encoders" in k) and isinstance(v, torch.Tensor) and v.is_floating_point():
            if vae:
                if k.startswith(vae):
                    continue
            theta[k] = v.to(target_dtype)
    return theta

cache_filename = os.path.join(os.getcwd(), "cache.json")

def _safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

def merge_cache_json(model_path):
    base = _safe_load_json(cache_filename)
    model_cache_path = os.path.join(model_path, "cache.json")
    update = _safe_load_json(model_cache_path) if os.path.exists(model_cache_path) else {}

    if isinstance(base, dict) and isinstance(update, dict):
        base.update(update)
    elif isinstance(base, list) and isinstance(update, list):
        base.extend(update)

    with filelock.FileLock(f"{cache_filename}.lock"):
        with open(cache_filename, "w", encoding="utf-8") as f:
            json.dump(base, f, ensure_ascii=False, indent=2)
    
def dump_cache(cache_data):
    with filelock.FileLock(f"{cache_filename}.lock"):
        with open(cache_filename, "w", encoding="utf8") as f:
            json.dump(cache_data or {}, f, indent=4)

def cache(subsection, cache_data):
    if cache_data is None:
        with filelock.FileLock(f"{cache_filename}.lock"):
            cache_data = _safe_load_json(cache_filename)
    if subsection not in cache_data or not isinstance(cache_data.get(subsection), dict):
        cache_data[subsection] = {}
        dump_cache(cache_data)
    return cache_data

def model_hash(filename: str) -> str:
    try:
        with open(filename, "rb") as f:
            f.seek(0x100000)
            return hashlib.sha256(f.read(0x10000)).hexdigest()[:8]
    except FileNotFoundError:
        return "NOFILE"

def sha256_from_cache(filename: str, title: str, cache_data):
    cache_data = cache("hashes", cache_data)
    hsect = cache_data.get("hashes", {})
    h = hsect.get(title)
    if not h:
        return None, None, cache_data
    return h.get("sha256"), h.get("model_hash"), cache_data

def calculate_sha256(filename: str, chunk_size: int = 4 * 1024 * 1024, max_workers: int = os.cpu_count() or 4) -> str:
    size = os.path.getsize(filename)
    n_chunks = (size + chunk_size - 1) // chunk_size
    hasher = hashlib.sha256()

    def _read(i):
        off = i * chunk_size
        with open(filename, "rb") as f:
            f.seek(off)
            return f.read(min(chunk_size, size - off))

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_read, i) for i in range(n_chunks)]
        for i in range(n_chunks):
            hasher.update(futures[i].result())
    return hasher.hexdigest()

def sha256(filename: str, title: str, cache_data=None) -> str:
    s256_cached, _, cache_data = sha256_from_cache(filename, title, cache_data)
    if s256_cached:
        return s256_cached

    print(f"Calculating sha256 for {filename}: ", end="")
    sha_val = calculate_sha256(filename)
    mhash = model_hash(filename)
    print(sha_val)

    cache_data = cache("hashes", cache_data)
    if "hashes" not in cache_data or not isinstance(cache_data["hashes"], dict):
        cache_data["hashes"] = {}

    cache_data["hashes"][title] = {
        "mtime": os.path.getmtime(filename) if os.path.exists(filename) else None,
        "sha256": sha_val,
        "model_hash": mhash,
    }
    dump_cache(cache_data)
    return sha_val

def calculate_shorthash(filename: str):
    title = f"checkpoint/{os.path.splitext(os.path.basename(filename))[0]}"
    val = sha256(filename, title, None)
    return None if val is None else val[:10]

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

def prune_model(theta, name, args, isxl=False, isflux=False, iszi=False):
    if not (isxl or isflux or iszi):
        _, _, auto_iszi, theta = detect_arch(theta)
        iszi = iszi or auto_iszi

    if isflux:
        cond_prefixes = ['clip.cond_stage_model.']
    elif isxl:
        cond_prefixes = ['conditioner.']
    elif iszi:
        cond_prefixes = ['text_encoders.qwen3_4b.']
    else:
        cond_prefixes = ['cond_stage_model.']

    roots = [
        'model.diffusion_model.',
        'depth_model.',
        'first_stage_model.',
        'vae.',
    ] + cond_prefixes
    
    theta_keys = list(theta.keys())

    for key in tqdm(theta_keys, desc=f"Pruning {name}..."):
        if not any(key.startswith(r) for r in roots):
            del theta[key]
            continue

        k_in = key
        if getattr(args, "keep_ema", False):
            k_ema = 'model_ema.' + key[6:].replace('.', '')
            if k_ema in theta:
                k_in = k_ema

        v = theta[k_in]
        if isinstance(v, torch.Tensor) and not (key.startswith("first_stage_model") or key.startswith("vae")):
            dt = v.dtype
            if getattr(args, "save_quarter", False) and dt in FP_SET:
                v = v.to(torch.float8_e4m3fn)
            elif getattr(args, "save_half", False) and dt in {torch.float32, torch.float64, torch.bfloat16, torch.float8_e4m3fn}:
                v = v.to(torch.float16)
            elif getattr(args, "save_bhalf", False) and dt in {torch.float32, torch.float64, torch.float16, torch.float8_e4m3fn}:
                v = v.to(torch.bfloat16)
            elif not getattr(args, "save_half", False) and not getattr(args, "save_bhalf", False) and dt in {torch.float16, torch.float64, torch.bfloat16, torch.float8_e4m3fn}:
                v = v.to(torch.float32)
        theta[key] = v

    return theta


def transform_checkpoint_dict_key(k: str):
    for src, rep in checkpoint_dict_replacements.items():
        if k.startswith(src):
            k = rep + k[len(src):]
    return k

def get_state_dict_from_checkpoint(pl_sd: dict) -> dict:
    d = pl_sd.pop("state_dict", pl_sd)
    d.pop("state_dict", None)
    out = {}
    for k, v in d.items():
        nk = transform_checkpoint_dict_key(k)
        if nk is not None and "model_sampling.sigmas" not in nk:
            out[nk] = v
    return out

def load_model(path: str, device, cache_data = None, verify_hash: bool = True):
    if path.endswith(".safetensors"):
        weights  = safetensors.torch.load_file(path, device=device)
        metadata = read_metadata_from_safetensors(path)
    else:
        weights  = torch.load(path, map_location=device)
        metadata = {}

    s256 = hashed = None
    if verify_hash:
        title = f"checkpoint/{Path(path).stem}"
        s256, hashed, cache_data = sha256_from_cache(path, title, cache_data)
        if not (s256 or hashed):
            sha256(path, title, cache_data)
            s256, hashed, cache_data = sha256_from_cache(path, title, cache_data)

    weights = get_state_dict_from_checkpoint(weights)
    if not verify_hash:
        metadata = None
    return weights, s256, hashed, metadata, cache_data

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

def maybe_to_qdtype(a, b, qa, qb, device, isflux):
    return to_qdtype(a, b, qa, qb, device) if isflux and qa != qb else (a, b)

def detect_arch(theta):
    isxl = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in theta
    isflux = any("double_block" in k for k in theta.keys())
    if "model.diffusion_model.cap_embedder.0.weight" in theta:
        iszi = True
    elif "cap_embedder.0.weight" in theta:
        new_key = {}
        for k in tqdm(theta.keys(), desc="Renaming Z-IMAGE keys..."):
            new_key["model.diffusion_model." + k.replace("model.diffusion_model.", "") if not (k.startswith("vae.") or k.startswith("text_encoders.")) else k] = theta[k]
        iszi = True
        theta = new_key
        del new_key
    return isxl, isflux, iszi, theta


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

def blocker(blocks: str, blockids: list[str]) -> str:
    out = []
    for w in blocks.split():
        if "-" in w:
            a, b = (t.strip() for t in w.split("-", 1))
            i, j = blockids.index(a), blockids.index(b)
            lo, hi = (i, j) if i <= j else (j, i)
            out.extend(blockids[lo:hi + 1])
        else:
            out.append(w)
    return " ".join(out)

def blockfromkey(key: str, isxl: bool = False, isflux: bool = False, iszi: bool = False) -> Tuple[str, str]:
    # SD1.5
    if not isxl and not isflux and not iszi:
        if "time_embed" in key: idx = -2
        elif ".out." in key:   idx = NUM_TOTAL_BLOCKS - 1
        elif (m := _re_inp.search(key)): idx = int(m.group(1))
        elif _re_mid.search(key):        idx = NUM_INPUT_BLOCKS
        elif (m := _re_out.search(key)): idx = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + int(m.group(1))
        else:                            return "Not Merge", "Not Merge"
        b = BLOCKID[idx + 1]
        return b, b

    # Flux
    if isflux:
        if "vae" in key:                 return "VAE",  "Not Merge"
        if "t5xxl" in key:               return "T5",   "T5"
        if "text_encoders.clip" in key:  return "CLIP", "CLIP"
        m = re.search(r'\.(\d+)\.', key)
        if "double_blocks" in key and m: return f"D{m.group(1).zfill(2)}", f"D{m.group(1).zfill(2)}"
        if "single_blocks" in key and m: return f"S{m.group(1).zfill(2)}", f"S{m.group(1).zfill(2)}"
        if "_in" in key:                 return "IN",   "IN"
        if "final_layer" in key:         return "OUT",  "OUT"
        return "Not Merge", "Not Merge"

    # SDXL
    if isxl:
        if not ("weight" in key or "bias" in key):     return "Not Merge", "Not Merge"
        if "label_emb" in key or "time_embed" in key:  return "Not Merge", "Not Merge"
        if "conditioner.embedders" in key:             return "BASE", "BASE"
        if "first_stage_model" in key:                 return "VAE",  "BASE"

        if "model.diffusion_model" in key:
            if "model.diffusion_model.out." in key:    return "OUT8", "OUT08"
            blk = (re.findall(r'input|mid|output', key) or [""])[0].upper().replace("PUT", "")
            nums = re.sub(r"\D", "", key)
            tag  = (nums[:1] + "0") if "MID" in blk else nums[:2]
            add  = (re.findall(r"transformer_blocks\.(\d+)\.", key) or [""])[0]
            left = blk + tag + add
            right = ("M00" if "MID" in blk else f"{blk}0{tag[0]}")
            return left, right
        
    #Z-IMAGE
    if iszi:
        if "qwen3_4b" in key or "cap_embedder" in key:          return "BASE", "BASE"
        if not ("weight" in key or "bias" in key):     return "Not Merge", "Not Merge"
        if "t_embedder" in key or "x_embedder" in key:     return "Not Merge", "Not Merge"
        if "vae" in key:                 return "VAE",  "VAE"
        if "norm_final" in key:          return "L29", "L29"
        
        if "model.diffusion_model" in key:
            if "model.diffusion_model.final_layer" in key:    return "L29", "L29"
            if "model.diffusion_model.context_refiner" in key:    return "CONT", "CONT"
            if "model.diffusion_model.noise_refiner" in key:    return "NOISE", "NOISE"
            m = (re.findall(r'layers\.(\d+)\.', key) or [""])[0]
            return (f"L{int(m):02}", f"L{int(m):02}") if m else ("Not Merge", "Not Merge")

    return "Not Merge", "Not Merge"

def elementals(key: str, weight_index: int, deep: list[str], current_alpha: float, blockids=BLOCKID) -> float:
    skey = key + blockids[weight_index]

    def _neg(tokens: list[str]):
        return (True, tokens[1:]) if tokens and tokens[0] == "NOT" else (False, tokens)

    for d in deep:
        if d.count(":") != 2:
            continue
        dbs_s, dws_s, dr_s = d.split(":", 2)

        dbs = blocker(dbs_s, blockids).split()
        dws = dws_s.split()
        dbn, dbs = _neg(dbs)
        dwn, dws = _neg(dws)

        ok = (any(db in skey for db in dbs) ^ dbn)
        if ok:
            ok = (any(dw in skey for dw in dws) ^ dwn)
        if ok:
            current_alpha = float(dr_s)

    return current_alpha

def prepare_merge_cache(theta_keys, isxl, isflux, iszi, deep_a, deep_b, weights_a, weights_b, alpha, beta):
    keymap = {}
    for k in tqdm(theta_keys, desc="Building merge cache..."):
        block, tag = blockfromkey(k, isxl, isflux, iszi)
        if block == "Not Merge": 
            continue
        if isflux and tag in BLOCKIDFLUX:
            wi = BLOCKIDFLUX.index(tag)
        elif isxl and tag in BLOCKIDXLL:
            wi = BLOCKIDXLL.index(tag)
        elif iszi and tag in BLOCKIDZI:
            wi = BLOCKIDZI.index(tag)
        elif tag in BLOCKID:
            wi = BLOCKID.index(tag)
        else:
            wi = -1
            
        blockids = BLOCKIDFLUX if isflux else (BLOCKIDXLL if isxl else (BLOCKIDZI if iszi else BLOCKID))

        cur_a = weights_a[wi - 1] if (weights_a is not None and wi > 0) else alpha
        cur_b = weights_b[wi - 1] if (weights_b is not None and wi > 0) else beta

        if deep_a:
            cur_a = elementals(k, wi, deep_a, cur_a, blockids)
        if deep_b:
            cur_b = elementals(k, wi, deep_b, cur_b, blockids)

        keymap[k] = (wi, cur_a, cur_b)
    return keymap

def diff_inplace(dst, src, func, desc):
    for k in tqdm(dst.keys(), desc=desc):
        if ("model" not in k) and ("text_encoders" not in k): 
            continue
        t2 = src.get(k, torch.zeros_like(dst[k])) if k in src else None
        dst[k] = func(dst[k], t2) if t2 is not None else torch.zeros_like(dst[k])

def clone_dict_tensors(d):
    return {k[0]: k[1].clone() for k in tqdm(list(d.items()), "Cloning dict...")}

def np_trim_percentiles(arr, lo=1, hi=99):
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return arr
    lo_v, hi_v = np.percentile(arr, lo, method='midpoint'), np.percentile(arr, hi, method='midpoint')
    return arr[(arr >= lo_v) & (arr <= hi_v)]

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
        "denoiser":"transformer", "denoise":"transformer", "transformer":"transformer", "mmdit":"transformer",
        "t5":"text", "t5-xxl":"text", "text1":"text", "text2":"text2",
        "all": "all",
    }
    mapped = [syn.get(t, t) for t in tokens]
    if "all" in mapped or not mapped:
        return {"unet", "vae", "clip-l", "clip-g", "clip", "transformer", "text", "text2"}
    return set(mapped)

def _component_prefix_map(isxl: bool, isflux: bool = False, iszi: bool = False):
    if isflux:
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

    if isxl:
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

    if iszi:
        return {
            "unet":        ["model.diffusion_model."],
            "transformer": ["model.diffusion_model."],
            "vae":         ["first_stage_model.", "vae."],
            "text":        ["qwen3_4b.", "cap_embedder.", "text_encoders.qwen3_4b."],
            "text2":       [],
            "clip":        ["qwen3_4b.", "cap_embedder.", "model.diffusion_model.cap_embedder.", "text_encoders.qwen3_4b."],
            "clip-l":      ["qwen3_4b.", "text_encoders.qwen3_4b."],
            "clip-g":      ["cap_embedder.", "model.diffusion_model.cap_embedder."],
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

def _swap_components_inplace(
    theta_dst: dict,
    theta_src: dict,
    components: set[str],
    isxl: bool,
    isflux: bool,
    iszi: bool = False,
):
    if not (isxl or isflux or iszi):
        _, _, auto_iszi, theta_src = detect_arch(theta_src)
        iszi = iszi or auto_iszi
    if iszi: print("Z-IMAGE architecture detected for component swapping.")

    pref = _component_prefix_map(isxl, isflux, iszi)
    selected = set()
    for c in components:
        if c in pref:
            selected.add(c)
        elif c in {"clip-l", "clip_g"} and "clip" in pref:
            selected.add(c)
        elif c == "clip" and "clip" in pref:
            selected.add("clip")

    prefixes = [p for c in selected for p in pref.get(c, [])]
    # print(prefixes)

    moved, created, skipped_shape = 0, 0, 0
    for k, v in theta_src.items():
        if not prefixes or _key_belongs_to_component(k, prefixes):
            if k in theta_dst and tuple(theta_dst[k].shape) != tuple(v.shape):
                skipped_shape += 1
                continue
            if k in theta_dst:
                theta_dst[k] = v.to(dtype=theta_dst[k].dtype, device=theta_dst[k].device)
                moved += 1
            else:
                theta_dst[k] = v
                created += 1
    return moved, created, skipped_shape, theta_dst


def _is_clip_key(key: str, isxl: bool, isflux: bool, iszi: bool = False) -> bool:
    if isflux:
        prefixes = ["text_encoder.", "text_encoder_2.", "conditioner.embedders.", "clip.", "t5."]
    elif isxl:
        prefixes = ["conditioner.embedders.", "text_encoders.", "clip_l.", "clip_g."]
    elif iszi:
        prefixes = ["qwen3_4b.", "cap_embedder.","text_encoders.qwen3_4b.","model.diffusion_model.cap_embedder."]
    else:
        prefixes = ["cond_stage_model.", "clip."]
    return any(key.startswith(p) for p in prefixes)


def _elemwise_union_minus_intersection(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    same_sign = torch.sign(a) == torch.sign(b)
    overlap = torch.where(
        same_sign,
        torch.sign(a) * torch.minimum(a.abs(), b.abs()),
        torch.zeros_like(a)
    )
    return a + b - overlap

def _elemwise_union_minus_intersection_with_base(base, A, B):
    dA, dB = A - base, B - base
    same_sign = torch.sign(dA) == torch.sign(dB)
    overlap = torch.where(
        same_sign,
        torch.sign(dA) * torch.minimum(dA.abs(), dB.abs()),
        torch.zeros_like(dA)
    )
    return base + (dA + dB - overlap)

def _projective_intersection_union(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a32 = a.detach().float().view(-1); b32 = b.detach().float().view(-1)
    if a32.numel() == 0:
        return a
    proj_b_on_a = (torch.dot(b32, a32) / (a32.norm()**2 + 1e-12)) * a32
    u = (a32 + b32 - proj_b_on_a).view_as(a)
    return u.to(a.dtype)

def _tensor_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a32 = a.detach().float().view(-1); b32 = b.detach().float().view(-1)
    if a32.numel() == 0:
        return torch.tensor(1.0, device=a.device)
    num = torch.dot(a32, b32)
    den = (a32.norm() * b32.norm()).clamp_min(1e-12)
    return (num / den).clamp(-1.0, 1.0)

def _cosine_gate(a: torch.Tensor, b: torch.Tensor, tau: float = 0.30, sharp: float = 0.10) -> torch.Tensor:
    cos = _tensor_cosine(a, b)
    g = torch.sigmoid((cos - float(tau)) / max(float(sharp), 1e-3))
    return g.to(a.dtype)

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

def _collect_clipxor_targets(theta_base: dict, theta_other: dict,
                             isxl: bool, isflux: bool, iszi: bool = False):
    targets = []
    for k, A in theta_base.items():
        if not _is_clip_key(k, isxl, isflux, iszi):
            continue
        B = theta_other.get(k)
        if getattr(A, "shape", None) != getattr(B, "shape", None):
            continue
        if _maybe_skip_small_norm_bias_for_clipxor(k, A):
            continue
        targets.append(k)
    return targets

def _clip_roots_for_arch(isxl: bool, isflux: bool, iszi: bool = False):
    if isflux:
        # Flux: T5 / text encoders
        return [
            "text_encoder.", "text_encoder_2.", "conditioner.embedders.", "clip.", "t5."
        ]
    if isxl:
        # SDXL: CLIP-L / CLIP-G
        return [
            "conditioner.embedders.0.", "conditioner.embedders.1.",
            "text_encoders.encoder_l.", "text_encoders.encoder_g.",
            "clip_l.", "clip_g."
        ]
    if iszi:
        # Z-Image: Qwen + caption embedder
        return [
            "qwen3_4b.",
            "cap_embedder.",
        ]
    # SD1.x / SD2.x (non-XL)
    return ["cond_stage_model.", "clip."]


def _iter_clip_items(sd: dict, isxl: bool, isflux: bool, iszi: bool = False):
    roots = _clip_roots_for_arch(isxl, isflux, iszi)
    for k, v in sd.items():
        for r in roots:
            if k.startswith(r):
                # canonical suffix after the first root occurrence
                suffix = k[len(r):]
                yield (k, r, suffix, v)
                break

def _collect_clip_pairs_by_suffix(sd_a: dict, sd_b: dict,
                                  isxl_a: bool, isflux_a: bool,
                                  isxl_b: bool, isflux_b: bool,
                                  iszi_a: bool = False, iszi_b: bool = False):
    # map suffix -> (orig_key, tensor) for A and B separately
    map_a = {}
    for k, root, suf, v in _iter_clip_items(sd_a, isxl_a, isflux_a, iszi_a):
        map_a[suf] = (k, v)
    map_b = {}
    for k, root, suf, v in _iter_clip_items(sd_b, isxl_b, isflux_b, iszi_b):
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
    if ("clip_l" in k) or ("text_model" in k and "clip" in k):
        return "clip-l"
    if ("clip_g" in k) or ("text2_model" in k and "clip" in k) or ("open_clip" in k):
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

def _norm_stats(t: torch.Tensor):
    v = t.detach().float().view(-1)
    if v.numel() == 0:
        return t.new_tensor(0.0), t.new_tensor(1.0)
    return v.mean(), v.std().clamp_min(1e-6)

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

def _finetune_inplace(key, tens, fine):
    if ("first_stage_model" in key or "vae" in key) or fine == "":
        return tens

    if isinstance(fine, dict):
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

    if isinstance(fine, list):
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


def update_model_a(ps: PermutationSpec, perm: dict, model_a: dict, new_alpha: float):
    for k in list(model_a.keys()):
        if k not in ps.axes_to_perm:
            continue
        try:
            perm_params = get_permuted_param(ps, perm, k, model_a)
            model_a[k] = model_a[k] * (1.0 - new_alpha) + new_alpha * perm_params
        except RuntimeError:
            continue
    return model_a

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