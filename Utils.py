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
from typing import List, Tuple
from pathlib import Path
import torch.nn.functional as F

FP_SET = {torch.float32, torch.float16, torch.float64, torch.bfloat16}

NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

BLOCKID = ["BASE"] + [f"IN{i:02}" for i in range(12)] + ["M00"] + [f"OUT{i:02}" for i in range(12)]
BLOCKIDXLL = ["BASE"] + [f"IN{i:02}" for i in range(9)] + ["M00"] + [f"OUT{i:02}" for i in range(9)] + ["VAE"]
BLOCKIDXL = ["BASE"] + [f"IN{i}" for i in range(9)] + ["M"] + [f"OUT{i}" for i in range(9)] + ["VAE"]
BLOCKIDFLUX = ["CLIP", "T5", "IN"] + ["D{:002}".format(x) for x in range(19)] + ["S{:002}".format(x) for x in range(38)] + ["OUT"] # Len: 61
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

file_path = os.path.join(os.getcwd(), "mbwpresets.txt")
if not os.path.isfile(file_path):
    shutil.copyfile(os.path.join(os.getcwd(), "mbwpresets_master.txt"), file_path)
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

def to_half_k(sd, enable):
    if enable:
        for d in tqdm(list(sd.items()), desc="Half tensoring..."):
            k, v = d
            if "model" in k and getattr(v, "dtype", None) in DTYPES:
                sd[k] = v.half()
    return sd

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

def prune_model(theta, name, args, isxl=False, isflux=False):
    condname = 'clip.cond_stage_model.' if isflux else ('conditioner.' if isxl else 'cond_stage_model.')
    sd_pruned = {}

    for key in tqdm(theta.keys(), desc=f"Pruning {name}..."):
        if not key.startswith(('model.diffusion_model.', 'depth_model.', 'first_stage_model.', condname)):
            continue

        k_in = key
        if getattr(args, "keep_ema", False):
            k_ema = 'model_ema.' + key[6:].replace('.', '')
            if k_ema in theta:
                k_in = k_ema

        v = theta[k_in]
        if isinstance(v, torch.Tensor):
            dt = v.dtype
            if getattr(args, "save_quarter", False) and dt in FP_SET:
                v = v.to(torch.float8_e4m3fn)
            elif getattr(args, "save_half", False) and dt in {torch.float32, torch.float64, torch.bfloat16}:
                v = v.to(torch.float16)
            elif not getattr(args, "save_half", False) and dt in {torch.float16, torch.float64, torch.bfloat16}:
                v = v.to(torch.float32)
        sd_pruned[key] = v

    return sd_pruned

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
        if nk is not None:
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
    return isxl, isflux

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

def blockfromkey(key: str, isxl: bool = False, isflux: bool = False):
    # SD1.5
    if not isxl and not isflux:
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

    return "Not Merge", "Not Merge"

def elementals(key: str, weight_index: int, deep: list[str], current_alpha: float):
    skey = key + BLOCKID[weight_index + 1]

    def _neg(tokens: list[str]):
        return (True, tokens[1:]) if tokens and tokens[0] == "NOT" else (False, tokens)

    for d in deep:
        if d.count(":") != 2:
            continue
        dbs_s, dws_s, dr_s = d.split(":", 2)

        dbs = blocker(dbs_s, BLOCKID).split()
        dws = dws_s.split()
        dbn, dbs = _neg(dbs)
        dwn, dws = _neg(dws)

        ok = (any(db in skey for db in dbs) ^ dbn)
        if ok:
            ok = (any(dw in skey for dw in dws) ^ dwn)
        if ok:
            current_alpha = float(dr_s)

    return current_alpha

def prepare_merge_cache(theta_keys, isxl, isflux, deep_a, deep_b, weights_a, weights_b, alpha, beta):
    keymap = {}
    for k in tqdm(theta_keys, desc="Building merge cache..."):
        block, tag = blockfromkey(k, isxl, isflux)
        if block == "Not Merge": 
            continue
        if isflux and tag in BLOCKIDFLUX:
            wi = BLOCKIDFLUX.index(tag)
        elif isxl and tag in BLOCKIDXLL:
            wi = BLOCKIDXLL.index(tag)
        elif tag in BLOCKID:
            wi = BLOCKID.index(tag)
        else:
            wi = -1

        cur_a = weights_a[wi - 1] if (weights_a is not None and wi > 0) else alpha
        cur_b = weights_b[wi - 1] if (weights_b is not None and wi > 0) else beta

        if deep_a:
            cur_a = elementals(k, wi, deep_a, cur_a)
        if deep_b:
            cur_b = elementals(k, wi, deep_b, cur_b)

        keymap[k] = (wi, cur_a, cur_b)
    return keymap

def diff_inplace(dst, src, func, desc):
    for k in tqdm(dst.keys(), desc=desc):
        if 'model' not in k: 
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

def _component_prefix_map(isxl: bool, isflux: bool = False):
    if isflux:
        return {
            "transformer": ["transformer."],
            "vae":         ["vae.", "first_stage_model."],
            "text":        ["text_encoder.", "conditioner.embedders.", "clip.", "t5."],
            "text2":       ["text_encoder_2."],
            "unet":        ["transformer."],
            "clip":        ["text_encoder.", "text_encoder_2.", "conditioner.embedders.", "clip."],
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
        }
    else:
        return {
            "unet": ["model.diffusion_model."],
            "vae":  ["first_stage_model."],
            "clip": ["cond_stage_model.", "clip."],
        }

def _key_belongs_to_component(key: str, prefixes: list[str]) -> bool:
    for p in prefixes:
        if key.startswith(p):
            return True
        if p.endswith(".") and key.startswith(p[:-1] + "/"):
            return True
    return False

def _swap_components_inplace(theta_dst: dict, theta_src: dict, components: set[str], isxl: bool, isflux: bool):
    pref = _component_prefix_map(isxl, isflux)
    selected = set()
    for c in components:
        if c in pref:
            selected.add(c)
        elif c in {"clip-l","clip-g"} and "clip" in pref:
            selected.add(c)
        elif c == "clip" and "clip" in pref:
            selected.add("clip")
    prefixes = [p for c in selected for p in pref.get(c, [])]

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

def _is_clip_key(key: str, isxl: bool, isflux: bool) -> bool:
    if isflux:
        prefixes = ["text_encoder.", "text_encoder_2.", "conditioner.embedders.", "clip.", "t5."]
    elif isxl:
        prefixes = ["conditioner.embedders.", "text_encoders.", "clip_l.", "clip_g."]
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

def _collect_clipxor_targets(theta_base: dict, theta_other: dict, isxl: bool, isflux: bool):
    targets = []
    for k, A in theta_base.items():
        if not _is_clip_key(k, isxl, isflux):
            continue
        B = theta_other.get(k)
        if getattr(A, "shape", None) != getattr(B, "shape", None):
            continue
        if _maybe_skip_small_norm_bias_for_clipxor(k, A):
            continue
        targets.append(k)
    return targets

def _clip_roots_for_arch(isxl: bool, isflux: bool):
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
    # SD1.x / SD2.x (non-XL)
    return ["cond_stage_model.", "clip."]

def _iter_clip_items(sd: dict, isxl: bool, isflux: bool):
    roots = _clip_roots_for_arch(isxl, isflux)
    for k, v in sd.items():
        for r in roots:
            if k.startswith(r):
                # canonical suffix after the first root occurrence
                suffix = k[len(r):]
                yield (k, r, suffix, v)
                break

def _collect_clip_pairs_by_suffix(sd_a: dict, sd_b: dict,
                                  isxl_a: bool, isflux_a: bool,
                                  isxl_b: bool, isflux_b: bool):
    # map suffix -> (orig_key, tensor) for A and B separately
    map_a = {}
    for k, root, suf, v in _iter_clip_items(sd_a, isxl_a, isflux_a):
        map_a[suf] = (k, v)
    map_b = {}
    for k, root, suf, v in _iter_clip_items(sd_b, isxl_b, isflux_b):
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
    if "first_stage_model" in key or not fine:
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