import os
import re
import json
import argparse
import torch
import torch.nn.functional as F
import safetensors.torch
from tqdm.autonotebook import tqdm

from Utils import (
    load_model,
    read_metadata_from_safetensors,
    sha256_from_cache,
    LBLOCKS26,
    LBLOCKS_FLUX,
    LBLOCKS_ZI,
    LBLOCKS_SDXL,
    BLOCKID,
    BLOCKIDFLUX,
    BLOCKIDZI,
    BLOCKIDXLL,
    cache,
    dump_cache,
    normalize_path,
    detect_arch,
    upcast_fp8_state_dict,
    prepare_state_dict_for_save,
    set_cache_filename,
    base_path,
    merge_cache_json,
)

_re_digits = re.compile(r"\d+")
_re_cache  = {}
_suffix_map = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2", "conv2": "out_layers_3",
        "norm1": "in_layers_0", "norm2": "out_layers_0",
        "time_emb_proj": "emb_layers_1", "conv_shortcut": "skip_connection",
    },
}

def _m(rx, key, buf):
    rg = _re_cache.get(rx) or re.compile(rx)
    _re_cache.setdefault(rx, rg)
    s = rg.match(key)
    if not s:
        return False
    buf[:] = [int(x) if _re_digits.fullmatch((x or "")) else x for x in s.groups()]
    return True

def convert_diffusers_name_to_compvis(key: str, is_sd2: bool) -> str:
    g: list = []
    if _m(r"lora_unet_conv_in(.*)", key, g):
        return f"diffusion_model_input_blocks_0_0{g[0]}"
    if _m(r"lora_unet_conv_out(.*)", key, g):
        return f"diffusion_model_out_2{g[0]}"
    if _m(r"lora_unet_time_embedding_linear_(\d+)(.*)", key, g):
        return f"diffusion_model_time_embed_{g[0]*2-2}{g[1]}"
    if _m(r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)", key, g):
        sfx = _suffix_map.get(g[1], {}).get(g[3], g[3])
        return f"diffusion_model_input_blocks_{1 + g[0]*3 + g[2]}_{1 if g[1]=='attentions' else 0}_{sfx}"
    if _m(r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)", key, g):
        sfx = _suffix_map.get(g[0], {}).get(g[2], g[2])
        return f"diffusion_model_middle_block_{1 if g[0]=='attentions' else g[1]*2}_{sfx}"
    if _m(r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)", key, g):
        sfx = _suffix_map.get(g[1], {}).get(g[3], g[3])
        return f"diffusion_model_output_blocks_{g[0]*3 + g[2]}_{1 if g[1]=='attentions' else 0}_{sfx}"
    if _m(r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv", key, g):
        return f"diffusion_model_input_blocks_{3 + g[0]*3}_0_op"
    if _m(r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv", key, g):
        return f"diffusion_model_output_blocks_{2 + g[0]*3}_{2 if g[0]>0 else 1}_conv"
    if _m(r"lora_unet_layers_(\d+)_(.+)", key, g):
        return f"diffusion_model_layers_{g[0]}_{g[1]}"
    if _m(r"lora_diffusion_model_layers_(\d+)_(.+)", key, g):
        return f"diffusion_model_layers_{g[0]}_{g[1]}"
    if _m(r"diffusion_model_layers_(\d+)_(.+)", key, g):
        return f"diffusion_model_layers_{g[0]}_{g[1]}"
    if _m(r"lora_unet_context_refiner_layers_(\d+)_(.+)", key, g):
        return f"diffusion_model_context_refiner_layers_{g[0]}_{g[1]}"
    if _m(r"lora_unet_noise_refiner_layers_(\d+)_(.+)", key, g):
        return f"diffusion_model_noise_refiner_layers_{g[0]}_{g[1]}"
    if _m(r"lora_te_text_model_encoder_layers_(\d+)_(.+)", key, g):
        if is_sd2:
            r = g[1].replace("mlp_fc1","mlp_c_fc").replace("mlp_fc2","mlp_c_proj").replace("self_attn","attn")
            return f"model_transformer_resblocks_{g[0]}_{r}"
        return f"transformer_text_model_encoder_layers_{g[0]}_{g[1]}"
    if _m(r"lora_te2_text_model_encoder_layers_(\d+)_(.+)", key, g):
        r = g[1].replace("mlp_fc1","mlp_c_fc").replace("mlp_fc2","mlp_c_proj").replace("self_attn","attn")
        return f"1_model_transformer_resblocks_{g[0]}_{r}"
    return key

_ZI_LAYER_RE = re.compile(
    r"^(?:model\.)?diffusion_model\.layers\.(\d+)\.(.+)\.lora_(A|down)\.weight$"
)

@torch.inference_mode()
def apply_lora_to_weight_inplace(W: torch.Tensor, up: torch.Tensor, down: torch.Tensor, scale: float, ratio: float):
    if W.ndim == 2:
        # W += ratio * (up @ down) * scale
        if not W.is_contiguous():
            W = W.contiguous()
        alpha = float(ratio) * float(scale)
        W.addmm_(up, down, beta=1.0, alpha=alpha)
        return W

    if down.size()[2:4] == (1, 1):
        u = up.squeeze(3).squeeze(2)
        d = down.squeeze(3).squeeze(2)
        delta = (u @ d).unsqueeze(2).unsqueeze(3)
        W.add_(delta, alpha=float(ratio) * float(scale))
        return W

    conved = F.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
    W.add_(conved, alpha=float(ratio) * float(scale))
    return W

def zimage_resolve_target(down_k: str):
    """
    return (target_weight_key_in_theta0, part) where part in {None,'q','k','v'}
    """
    m = _ZI_LAYER_RE.match(down_k)
    if not m:
        return None, None

    layer = int(m.group(1))
    tail  = m.group(2)  # e.g. "attention.to_q" / "feed_forward.w1" / "adaLN_modulation.0" etc

    part = None

    # attention mapping
    if tail.startswith("attention.to_out.0"):
        # to_out.0 -> out
        tgt_tail = tail.replace("attention.to_out.0", "attention.out")
    elif tail.startswith("attention.to_q"):
        tgt_tail = tail.replace("attention.to_q", "attention.qkv")
        part = "q"
    elif tail.startswith("attention.to_k"):
        tgt_tail = tail.replace("attention.to_k", "attention.qkv")
        part = "k"
    elif tail.startswith("attention.to_v"):
        tgt_tail = tail.replace("attention.to_v", "attention.qkv")
        part = "v"
    else:
        tgt_tail = tail

    target = f"model.diffusion_model.layers.{layer}.{tgt_tail}.weight"
    return target, part

@torch.inference_mode()
def apply_zimage_lora(
    theta_0: dict,
    target_key: str,
    part: str | None,
    up: torch.Tensor,
    down: torch.Tensor,
    alpha,
    ratio: float,
):
    W = theta_0.get(target_key)
    if W is None or not isinstance(W, torch.Tensor):
        return False

    rank = int(down.size(0))
    a = alpha
    if isinstance(a, torch.Tensor):
        a = float(a.item())
    a = float(rank if a is None else a)

    scale = (a / float(rank))
    alpha_mm = float(ratio) * scale

    orig_dtype = W.dtype
    compute_dtype = (torch.float32 if (W.device.type == "cpu" and orig_dtype != torch.float32) else orig_dtype)

    Wc = W.to(dtype=compute_dtype)
    uc = up.to(device=W.device, dtype=compute_dtype)
    dc = down.to(device=W.device, dtype=compute_dtype)

    if Wc.ndim != 2:
        apply_lora_to_weight_inplace(Wc, uc, dc, scale, ratio=float(ratio))
    else:
        if part is None:
            Wc.addmm_(uc, dc, beta=1.0, alpha=alpha_mm)
        else:
            d = uc.shape[0]
            off = {"q": 0, "k": d, "v": 2 * d}[part]
            view = Wc.narrow(0, off, d)
            view.addmm_(uc, dc, beta=1.0, alpha=alpha_mm)

    if compute_dtype != orig_dtype:
        Wc = Wc.to(orig_dtype)

    theta_0[target_key] = Wc
    return True


def load_state_dict(path: str, dtype=torch.float, device="cpu", depatch=True):
    if path.endswith(".safetensors"):
        sd = safetensors.torch.load_file(path, device=device)
        meta = _safe_meta(path)
    else:
        sd = torch.load(path, map_location=device)
        meta = {}
    isv2 = any("resblocks" in k for k in sd.keys())
    if depatch:
        for k, v in list(sd.items()):
            if isinstance(v, torch.Tensor):
                sd[k] = v.to(dtype=dtype, device=device)
    return sd, meta, isv2

def _safe_meta(st_path: str) -> dict:
    try:
        with safetensors.safe_open(st_path, framework="pt", device="cpu") as f:
            return f.metadata() or {}
    except Exception:
        return {}

def apply_dare(delta: torch.Tensor, p: float):
    m = torch.bernoulli(torch.full(delta.shape, p, device=delta.device, dtype=delta.dtype))
    return (m * delta) / (1 - p)

def _l2(x, eps=1e-12): return x / (x.norm() + eps)

def spectral_norm(W: torch.Tensor, it=10):
    u = torch.randn(1, W.size(0), device=W.device, dtype=torch.float32)
    w = W.to(u.device, dtype=torch.float32)
    for _ in range(max(1, it)):
        v = _l2(u @ w.view(u.shape[-1], -1))
        u = _l2(v @ w.view(u.shape[-1], -1).t())
    return (u @ w.view(u.shape[-1], -1) @ v.t()).sum().item()

def apply_spectral_norm(lora_sd: dict, scale: float):
    lips = [spectral_norm(t) for k, t in lora_sd.items() if "alpha" not in k]
    if not lips: return lora_sd
    s = max(lips)
    if s <= 0:   return lora_sd
    fac = scale / s
    for k, t in lora_sd.items():
        if "alpha" not in k:
            lora_sd[k] = t * fac
    return lora_sd

@torch.inference_mode()
def merge_weights_inplace(
    lora: dict,
    isv2: bool,
    isxl: bool,
    blocks: list[list[str]],
    p: float,
    lam: float,
    scale: float,
    strengths: list[float],
    *,
    spectral_it: int = 2,
):
    def _pick_strength(full: str, msd: str):
        s0 = strengths[0] if strengths else 1.0
        for i, b in enumerate(blocks):
            for alias in b:
                if (alias in full) or (alias in msd):
                    return strengths[i] if i < len(strengths) else s0
        return s0

    def _apply_dare_inplace(t: torch.Tensor):
        if p <= 0:
            return t
        m = (torch.rand_like(t) < p).to(t.dtype)
        return (m * t) / (1.0 - p)

    for k, v in list(lora.items()):
        if "alpha" in k:
            continue
        full = convert_diffusers_name_to_compvis(k, isv2)
        msd  = full.split(".", 1)[0]
        if isxl:
            msd = msd.replace("lora_unet", "diffusion_model").replace("lora_te1_text_model", "0_transformer_text_model")

        strength = _pick_strength(full, msd)
        lora[k] = (strength * lam) * _apply_dare_inplace(v)

    if scale > 0:
        def _l2(x, eps=1e-12): return x / (x.norm() + eps)

        def spectral_norm_fast(W: torch.Tensor, it=spectral_it):
            u = torch.randn(1, W.size(0), device=W.device, dtype=torch.float32)
            w = W.to(u.device, dtype=torch.float32)
            for _ in range(max(1, it)):
                v = _l2(u @ w.view(u.shape[-1], -1))
                u = _l2(v @ w.view(u.shape[-1], -1).t())
            return (u @ w.view(u.shape[-1], -1) @ v.t()).sum().item()

        lips = []
        for kk, tt in lora.items():
            if "alpha" in kk:
                continue
            lips.append(spectral_norm_fast(tt))
        if lips:
            s = max(lips)
            if s > 0:
                fac = float(scale) / float(s)
                for kk, tt in lora.items():
                    if "alpha" not in kk:
                        lora[kk] = tt * fac

    return lora

@torch.inference_mode()
def build_apply_plan(main_keys, *, isxl: bool, iszi: bool, mlv2: bool, keymap: dict):
    plan = []
    for k in main_keys:
        down_k, up_k, alpha_k = parse_lora_key(k)
        if down_k is None:
            continue

        if iszi:
            tgt, part = zimage_resolve_target(down_k)
            if tgt is not None:
                plan.append(("zi", down_k, up_k, alpha_k, tgt, part))
            continue

        full = convert_diffusers_name_to_compvis(down_k, mlv2)
        msd  = full.split(".", 1)[0]
        if isxl:
            msd = msd.replace("lora_unet", "diffusion_model").replace("lora_te1_text_model", "0_transformer_text_model")

        wkey = keymap.get(msd)
        if wkey is None:
            continue
        plan.append(("std", down_k, up_k, alpha_k, wkey, None))
    return plan

def get_loralist(arg: str):
    return [x.split(":", 1) if ":" in x else [x, "1.0"] for x in arg.split(",") if x.strip()]

def _build_keymap(sd: dict):
    km = {}
    for k in sd.keys():
        if ("model" not in k) and ("text_encoders" not in k) and ("vae" not in k) and ("first_stage_model" not in k): 
            continue
        sk = k.replace(".", "_").replace("_weight", "")
        if "conditioner_embedders_" in sk:
            km[sk.split("conditioner_embedders_", 1)[1]] = k
        elif "wrapped_" in sk:
            km[sk.split("wrapped_", 1)[1]] = k
        elif "clip_l" in sk or "t5xxl" in sk:
            parts = sk.split("text_encoders_", 1)
            if len(parts) == 2: km[parts[1]] = k
        elif "model_" in sk:
            km[sk.split("model_", 1)[1]] = k
    return km


def _normalize_blocks(blocks):
    out = []
    for b in blocks or []:
        out.append([b] if isinstance(b, str) else list(b))
    return out

def _find_block_index(full: str, msd: str, blocks_norm: list[list[str]]) -> int:
    for i, aliases in enumerate(blocks_norm):
        for a in aliases:
            if (a in full) or (a in msd):
                return i
    return 0

def _iter_lora_down_keys(sd: dict):
    for k in sd.keys():
        if ("lora_A" in k) or ("lora_down" in k):
            yield k

def _pair_from_down_key(down_k: str):
    if "lora_A" in down_k:
        up_k    = down_k.replace("lora_A", "lora_B")
        alpha_k = down_k.replace("lora_A", "alpha")
        return down_k, up_k, alpha_k
    if "lora_down" in down_k:
        up_k    = down_k.replace("lora_down", "lora_up")
        alpha_k = down_k.replace("lora_down", "alpha")
        return down_k, up_k, alpha_k
    return None, None, None

_conv_cache = {}  # (is_sd2, key) -> full

def convert_diffusers_name_to_compvis_cached(key: str, is_sd2: bool) -> str:
    ck = (bool(is_sd2), key)
    v = _conv_cache.get(ck)
    if v is not None:
        return v
    v = convert_diffusers_name_to_compvis(key, is_sd2)
    if len(_conv_cache) > 200000:
        _conv_cache.clear()
    _conv_cache[ck] = v
    return v

def parse_lora_key(k: str):
    if "lora_A" in k:
        down = k
        up   = k.replace("lora_A", "lora_B")
        alpha = k.replace("lora_A", "alpha")
        return down, up, alpha

    if "lora_down" in k:
        down = k
        up   = k.replace("lora_down", "lora_up")
        alpha = k.replace("lora_down", "alpha")
        return down, up, alpha

    return None, None, None


@torch.inference_mode()
def pluslora(lora_list, model, output, model_path, device="cpu"):
    set_cache_filename(os.path.join(base_path(), "cache.json"))
    merge_cache_json(model_path)
    cache_data = cache("hashes", None)
    model_path = normalize_path(model_path)
    if not model:     return "ERROR: No model Selected"
    if not lora_list: return "ERROR: No LoRA Selected"

    print("Plus LoRA start")

    # checkpoint
    mpath = normalize_path(os.path.join(model_path, model))
    theta_0, *_ = load_model(mpath, device)
    theta_0 = upcast_fp8_state_dict(theta_0)
    model_name  = os.path.splitext(os.path.basename(mpath))[0]

    isxl, isflux, iszi, theta_0 = detect_arch(theta_0)

    blocks  = LBLOCKS_ZI if iszi else (LBLOCKS_FLUX if isflux else (LBLOCKS_SDXL if isxl else LBLOCKS26))
    blocksN = _normalize_blocks(blocks)

    blocknum = BLOCKIDZI if iszi else (BLOCKIDFLUX if isflux else (BLOCKIDXLL if isxl else BLOCKID))
    vae_key  = "first_stage_model" if not (isflux or iszi) else "vae"

    keymap   = _build_keymap(theta_0)

    lr_strs   = []
    lora_meta = {}

    for lora_model, ratio_str in lora_list:
        print(f"loading: {lora_model}")

        ratios = ([float(x) for x in ratio_str.replace(" ", "").split(",")]
                  if isinstance(ratio_str, str) else [float(ratio_str)] * len(blocknum))
        lr_strs.append("[" + ",".join(str(x) for x in ratios) + "]")

        lpath = normalize_path(os.path.join(model_path, lora_model))
        lsd, meta, lisv2 = load_state_dict(lpath, torch.float, depatch=False)
        lhash, _, cache_data = sha256_from_cache(
            lpath, f"lora/{os.path.splitext(os.path.basename(lpath))[0]}", cache_data
        )
        lora_meta[lhash] = meta

        # --- apply plan build ---
        plan = []  # (kind, target_key, part, down_k, up_k, alpha_k, ratio)
        for down_k in tqdm(list(_iter_lora_down_keys(lsd)), desc=f"Planning {lora_model}...", leave=False):
            d, u, a = _pair_from_down_key(down_k)
            if d is None:
                continue
            if (u not in lsd) or (d not in lsd):
                continue
            
            if iszi:
                target_key, part = zimage_resolve_target(d)
                if target_key is None:
                    continue
                full = convert_diffusers_name_to_compvis_cached(d, lisv2)
                msd  = full.split(".", 1)[0]
                bi   = _find_block_index(full, msd, blocksN)
                ratio = ratios[bi] if bi < len(ratios) else ratios[0]
                plan.append(("zi", target_key, part, d, u, a, float(ratio)))
                continue

            # standard (SD/SDXL/Flux)
            full = convert_diffusers_name_to_compvis_cached(d, lisv2)
            msd  = full.split(".", 1)[0]
            if isxl:
                msd = msd.replace("lora_unet","diffusion_model").replace("lora_te1_text_model","0_transformer_text_model")

            target = keymap.get(msd)
            if target is None:
                continue

            bi    = _find_block_index(full, msd, blocksN)
            ratio = ratios[bi] if bi < len(ratios) else ratios[0]
            plan.append(("std", target, None, d, u, a, float(ratio)))

        # --- apply ---
        for kind, target_key, part, down_k, up_k, alpha_k, ratio in tqdm(plan, desc=f"Merging {lora_model}...", leave=False):
            if target_key not in theta_0:
                continue

            # alpha / scale
            down = lsd[down_k]
            up   = lsd[up_k]
            dim  = int(down.size(0))
            alpha = lsd.get(alpha_k, dim)
            if alpha is None and isinstance(alpha_k, str) and alpha_k.endswith(".weight"):
                alpha = lsd.get(alpha_k[:-len(".weight")], dim)

            sc = float(alpha) / float(dim)

            if kind == "zi":
                apply_zimage_lora(
                    theta_0, target_key, part,
                    up=up, down=down,
                    alpha=alpha,
                    ratio=ratio
                )
                continue

            W = theta_0[target_key]
            dev = W.device
            dt  = W.dtype if (isinstance(W, torch.Tensor) and W.is_floating_point()) else torch.float32

            u = up.to(device=dev, dtype=dt, non_blocking=False)
            d = down.to(device=dev, dtype=dt, non_blocking=False)

            theta_0[target_key] = apply_lora_to_weight_inplace(W, u, d, sc, ratio)

        del lsd
        
    prepare_state_dict_for_save(
        theta_0, args,
        isxl=isxl, isflux=isflux, iszi=iszi,
        vae_prefix=vae_key,
        prune=bool(getattr(args, "prune", False)),
        make_cpu=True,
        make_contiguous=True,
    )

    out_name = os.path.splitext(os.path.basename(output))[0]
    meta_new = {
        "sd_merge_models": json.dumps({
            "type": "pluslora-chattiori",
            "checkpoint_hash": sha256_from_cache(mpath, f"checkpoint/{model_name}", cache_data)[0],
            "lora_hash": ",".join([k for k in lora_meta.keys() if k]),
            "alpha_info": ",".join(lr_strs),
            "output_name": out_name,
        }),
        "checkpoint": json.dumps(read_metadata_from_safetensors(mpath)) if mpath.endswith(".safetensors") else "{}",
        "lora": json.dumps(lora_meta),
    }
    if args.memo is not None:
        meta_new["memo"] = args.memo

    print(f"Saving as {output}...")
    if output.endswith(".safetensors"):
        safetensors.torch.save_file(theta_0, output, metadata=None if args.no_metadata else meta_new)
    else:
        torch.save({"state_dict": theta_0}, output)

    del theta_0
    dump_cache(cache_data)
    print(f"Done! ({round(os.path.getsize(output)/1073741824, 2)}G)")


@torch.inference_mode()
def darelora(mainlora, lora_list, model, output, model_path, device="cpu"):
    set_cache_filename(os.path.join(base_path(), "cache.json"))
    merge_cache_json(model_path)
    cache_data = cache("hashes", None)
    model_path = normalize_path(model_path)
    if not model:     return "ERROR: No model Selected"
    if not lora_list: return "ERROR: No LoRA Selected"

    print("Plus LoRA DARE start")

    # checkpoint
    mpath = normalize_path(os.path.join(model_path, model))
    theta_0, *_ = load_model(mpath, device)
    theta_0 = upcast_fp8_state_dict(theta_0)
    model_name  = os.path.splitext(os.path.basename(mpath))[0]

    isxl, isflux, iszi, theta_0 = detect_arch(theta_0)
    blocks  = LBLOCKS_ZI if iszi else (LBLOCKS_FLUX if isflux else (LBLOCKS_SDXL if isxl else LBLOCKS26))
    blocknum = BLOCKIDZI if iszi else (BLOCKIDFLUX if isflux else (BLOCKIDXLL if isxl else BLOCKID))
    vae_key = "first_stage_model" if not (isflux or iszi) else "vae"

    keymap = _build_keymap(theta_0)

    # mainlora
    main_sd, _, mlv2 = load_state_dict(mainlora, torch.float, depatch=False)
    plan = build_apply_plan(main_sd.keys(), isxl=isxl, iszi=iszi, mlv2=mlv2, keymap=keymap)
    del main_sd

    # DARE params
    lam, p, scale = 1.5, 0.13, 0.2
    torch.manual_seed(0)

    lr_strs, lora_meta = [], {}

    for lora_model, ratio_str in lora_list:
        print(f"loading: {lora_model}")

        ratios = ([float(x) for x in ratio_str.replace(" ", "").split(",")]
                  if isinstance(ratio_str, str) else [ratio_str] * len(blocknum))
        lr_strs.append("[" + ",".join(str(x) for x in ratios) + "]")

        lpath = normalize_path(os.path.join(model_path, lora_model))
        lsd, meta, lisv2 = load_state_dict(lpath, torch.float, depatch=False)
        lhash, _, cache_data = sha256_from_cache(lpath, f"lora/{os.path.splitext(os.path.basename(lpath))[0]}", cache_data)
        lora_meta[lhash] = meta

        lw = merge_weights_inplace(lsd, lisv2, isxl, blocks, p, lam, scale, ratios)

        for kind, down_k, up_k, alpha_k, tgt, part in tqdm(plan, desc=f"Applying {lora_model}...", leave=False):
            if (down_k not in lw) or (up_k not in lw):
                continue

            # alpha / scale
            down = lw[down_k]
            up   = lw[up_k]
            dim  = int(down.size(0))
            alpha = lw.get(alpha_k, dim)
            sc = float(alpha) / float(dim)

            if kind == "zi":
                apply_zimage_lora(
                    theta_0, tgt, part,
                    up=up, down=down,
                    alpha=alpha,
                    ratio=1.0,
                )
                continue

            if tgt not in theta_0:
                continue
            W = theta_0[tgt]

            dev = W.device
            W32 = W.float()
            out = apply_lora_to_weight_inplace(W32, up.to(dev).float(), down.to(dev).float(), sc, ratio=1.0)
            theta_0[tgt] = out.to(W.dtype)

        del lw, lsd
        
    prepare_state_dict_for_save(
        theta_0, args,
        isxl=isxl, isflux=isflux, iszi=iszi,
        vae_prefix=vae_key,
        prune=bool(getattr(args, "prune", False)),
        make_cpu=True,
        make_contiguous=True,
    )

    out_name = os.path.splitext(os.path.basename(output))[0]
    meta_new = {
        "sd_merge_models": json.dumps({
            "type": "pluslora-chattiori",
            "checkpoint_hash": sha256_from_cache(mpath, f"checkpoint/{model_name}", cache_data)[0],
            "lora_hash": ",".join([k for k in lora_meta.keys() if k]),
            "alpha_info": "DARE:" + ",".join(lr_strs),
            "output_name": out_name,
        }),
        "checkpoint": json.dumps(read_metadata_from_safetensors(mpath)) if mpath.endswith(".safetensors") else "{}",
        "lora": json.dumps(lora_meta),
    }
    if args.memo is not None:
        meta_new["memo"] = args.memo

    print(f"Saving as {output}...")
    if output.endswith(".safetensors"):
        safetensors.torch.save_file(theta_0, output, metadata=None if args.no_metadata else meta_new)
    else:
        torch.save({"state_dict": theta_0}, output)

    del theta_0
    dump_cache(cache_data)
    print(f"Done! ({round(os.path.getsize(output)/1073741824, 2)}G)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge several loras to checkpoint")
    parser.add_argument("model_path", type=str, help="Path to models")
    parser.add_argument("checkpoint", type=str, help="Name of the checkpoint")
    parser.add_argument("loras", type=str, help="Path and alpha of LoRAs eg.)\"Path:alpha,Path:alpha, ...\"")
    parser.add_argument("--save_half", action="store_true", help="Save as float16", required=False)
    parser.add_argument("--save_bhalf", action="store_true", help="Save as bfloat16", required=False)
    parser.add_argument("--prune", action="store_true", help="Prune Model", required=False)
    parser.add_argument("--save_quarter", action="store_true", help="Save as float8", required=False)
    parser.add_argument("--keep_ema", action="store_true", help="Keep ema", required=False)
    parser.add_argument("--dare", action="store_true", help="Use DARE Merge")
    parser.add_argument("--no_metadata", action="store_true", help="Save without metadata")
    parser.add_argument("--memo",   type=str,   help="Additional info bake in metadata", default=None)
    parser.add_argument("--save_safetensors", action="store_true", help="Save as .safetensors", required=False)
    parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
    parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
    args = parser.parse_args()
    args.model_path = normalize_path(args.model_path)

    ll  = get_loralist(args.loras)
    out = normalize_path(os.path.join(args.model_path, f"{args.output}.{'safetensors' if args.save_safetensors else 'ckpt'}"))

    if args.dare:
        mainlora = normalize_path(os.path.join(args.model_path, ll[0][0]))
        darelora(mainlora, ll, args.checkpoint, out, args.model_path, args.device)
    else:
        pluslora(ll, args.checkpoint, out, args.model_path, args.device)