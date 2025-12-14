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
    prune_model,
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
    to_quarter_k,
    to_half_k,
    upcast_fp8_state_dict,
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

def apply_zimage_lora(theta_0: dict, target_key: str, part: str | None,
                      up: torch.Tensor, down: torch.Tensor,
                      alpha, ratio: float):
    if target_key not in theta_0:
        return False

    W = theta_0[target_key]
    orig_dtype = W.dtype

    W32 = W.to(torch.float32)
    up32 = up.to(torch.float32)
    down32 = down.to(torch.float32)

    rank = down.size(0)
    a = alpha
    if isinstance(a, torch.Tensor):
        a = a.item()
    if a is None:
        a = rank
    sc = float(a) / float(rank)

    if W32.ndim != 2:
        Wnew = _apply_lora_to_weight(W32, up32, down32, sc, ratio)
        theta_0[target_key] = Wnew.to(orig_dtype)
        return True

    delta = (up32 @ down32) * (sc * ratio)

    if part is None:
        W32 = W32 + delta
    else:
        d = delta.shape[0]                 # = hidden_dim
        off = {"q": 0, "k": d, "v": 2*d}[part]
        W32[off:off+d, :] += delta

    theta_0[target_key] = W32.to(orig_dtype)
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

def merge_weights(lora: dict, isv2: bool, isxl: bool, blocks: list[str], p: float, lam: float, scale: float, strengths: list[float]):
    out = {}
    for k, v in lora.items():
        if "alpha" in k:
            out[k] = v
            continue
        full = convert_diffusers_name_to_compvis(k, isv2)
        msd  = full.split(".", 1)[0]
        if isxl:
            msd = msd.replace("lora_unet", "diffusion_model").replace("lora_te1_text_model", "0_transformer_text_model")
        strength = strengths[0]
        for i, b in enumerate(blocks):
            if any(b[k] in full or b[k] in msd for k in range(len(b))):
                strength = strengths[i] if i < len(strengths) else strengths[0]
                break
        out[k] = strength * lam * apply_dare(v, p)
    return apply_spectral_norm(out, scale) if scale > 0 else out

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

def _apply_lora_to_weight(W: torch.Tensor, up: torch.Tensor, down: torch.Tensor, scale: float, ratio: float):
    if W.ndim == 2:
        return W + ratio * (up @ down) * scale
    if down.size()[2:4] == (1, 1):
        u = up.squeeze(3).squeeze(2); d = down.squeeze(3).squeeze(2)
        return W + ratio * (u @ d).unsqueeze(2).unsqueeze(3) * scale
    conved = F.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
    return W + ratio * conved * scale

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

def pluslora(lora_list, model, output, model_path, device="cpu"):
    cache_data = cache("hashes", None)
    model_path = normalize_path(model_path)
    if not model:     return "ERROR: No model Selected"
    if not lora_list: return "ERROR: No LoRA Selected"

    print("Plus LoRA start")
    mpath = normalize_path(os.path.join(model_path, model))
    theta_0, *_ = load_model(mpath, device)
    theta_0 = upcast_fp8_state_dict(theta_0)
    model_name  = os.path.splitext(os.path.basename(mpath))[0]

    isxl, isflux, iszi, theta_0 = detect_arch(theta_0)
    blocks = LBLOCKS_ZI if iszi else (LBLOCKS_FLUX if isflux else (LBLOCKS_SDXL if isxl else LBLOCKS26))
    blocknum = BLOCKIDZI if iszi else (BLOCKIDFLUX if isflux else (BLOCKIDXLL if isxl else BLOCKID))
    vae_key = "first_stage_model" if not (isflux or iszi) else "vae"

    keymap   = _build_keymap(theta_0)
    lr_strs  = []
    lora_meta = {}

    for lora_model, ratio_str in lora_list:
        print(f"loading: {lora_model}")
        ratios = ([float(x) for x in ratio_str.replace(" ", "").split(",")] 
                  if isinstance(ratio_str, str) else [ratio_str] * len(blocknum))
        lr_strs.append("[" + ",".join(str(x) for x in ratios) + "]")
        lpath = normalize_path(os.path.join(model_path, lora_model))
        lsd, meta, lisv2 = load_state_dict(lpath, torch.float)
        lhash, _, cache_data = sha256_from_cache(lpath, f"lora/{os.path.splitext(os.path.basename(lpath))[0]}", cache_data)
        lora_meta[lhash] = meta
        

        for k in tqdm(list(lsd.keys()), desc=f"Merging {lora_model}..."):
            down_k, up_k, alpha_k = parse_lora_key(k)
            if down_k is None:
                continue

            if down_k not in lsd or up_k not in lsd:
                continue

            full = convert_diffusers_name_to_compvis(k, lisv2)
            msd  = full.split(".", 1)[0]
            
            if isxl:
                msd = msd.replace("lora_unet","diffusion_model").replace("lora_te1_text_model","0_transformer_text_model")
            
            ratio = ratios[0]
            for i, b in enumerate(blocks):
                if any(b[k] in full or b[k] in msd for k in range(len(b))):
                    ratio = ratios[i] if i < len(ratios) else ratios[0]
                    break
                
            if iszi:
                target_key, part = zimage_resolve_target(down_k)
                if target_key is not None:
                    rank = lsd[down_k].size(0)
                    alpha = lsd.get(alpha_k, None)
                    if alpha is None and isinstance(alpha_k, str) and alpha_k.endswith(".weight"):
                        alpha = lsd.get(alpha_k[:-len(".weight")], None)

                    ok = apply_zimage_lora(theta_0, target_key, part,
                                        up=lsd[up_k], down=lsd[down_k],
                                        alpha=(alpha if alpha is not None else rank),
                                        ratio=ratio)
                    if ok:
                        continue
            if msd not in keymap: 
                continue
                
            down = lsd[down_k].to(device)
            up   = lsd[up_k].to(device)

            dim   = down.size(0)
            alpha = lsd.get(alpha_k, dim)
            sc    = (alpha / dim)

            W = theta_0[keymap[msd]].to(device)
            theta_0[keymap[msd]] = torch.nn.Parameter(
                _apply_lora_to_weight(W, up, down, sc, ratio)
            )
        del lsd
        
    theta_0 = to_half_k(theta_0, args.save_half, args.save_bhalf, vae=vae_key)

    if args.prune:
        theta_0 = prune_model(theta_0, "Model", args, isxl=isxl, isflux=isflux, iszi=iszi)

    theta_0 = to_quarter_k(theta_0, args.save_quarter, prefer="e4m3", vae=vae_key)

    for k in tqdm(list(theta_0.keys()), desc="Check contiguous..."):
        if isinstance(theta_0[k], torch.Tensor):
            theta_0[k] = theta_0[k].detach().cpu().contiguous()

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

    print(f"Saving as {output}...")
    if output.endswith(".safetensors"):
        safetensors.torch.save_file(theta_0, output, metadata=meta_new)
    else:
        torch.save({"state_dict": theta_0}, output)

    del theta_0
    dump_cache(cache_data)
    print(f"Done! ({round(os.path.getsize(output)/1073741824, 2)}G)")

def darelora(mainlora, lora_list, model, output, model_path, device="cpu"):
    cache_data = cache("hashes", None)
    model_path = normalize_path(model_path)
    if not model:     return "ERROR: No model Selected"
    if not lora_list: return "ERROR: No LoRA Selected"

    print("Plus LoRA DARE start")
    mpath = normalize_path(os.path.join(model_path, model))
    theta_0, *_ = load_model(mpath, device)
    theta_0 = upcast_fp8_state_dict(theta_0)
    model_name  = os.path.splitext(os.path.basename(mpath))[0]

    isxl, isflux, iszi, theta_0 = detect_arch(theta_0)
    blocks = LBLOCKS_ZI if iszi else (LBLOCKS_FLUX if isflux else (LBLOCKS_SDXL if isxl else LBLOCKS26))
    blocknum = BLOCKIDZI if iszi else (BLOCKIDFLUX if isflux else (BLOCKIDXLL if isxl else BLOCKID))
    vae_key = "first_stage_model" if not (isflux or iszi) else "vae"
    keymap = _build_keymap(theta_0)

    main_sd, _, mlv2 = load_state_dict(mainlora, torch.float, depatch=False)

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

        lw = merge_weights(lsd, lisv2, isxl, blocks, p, lam, scale, ratios)

        for k in tqdm(list(main_sd.keys()), desc=f"Merging {lora_model}..."):
            down_k, up_k, alpha_k = parse_lora_key(k)
            if down_k is None:
                continue
            if down_k not in lw or up_k not in lw:
                continue
            
            full = convert_diffusers_name_to_compvis(down_k, mlv2)
            msd  = full.split(".", 1)[0]
            if isxl:
                msd = msd.replace("lora_unet","diffusion_model").replace("lora_te1_text_model","0_transformer_text_model")
            if msd not in keymap:
                continue

            down = lw[down_k].to(device)
            up   = lw[up_k].to(device)
            dim  = down.size(0)
            alpha = lw.get(alpha_k, dim)
            sc    = alpha / dim

            W = theta_0[keymap[msd]].to(device)
            theta_0[keymap[msd]] = torch.nn.Parameter(
                _apply_lora_to_weight(W, up, down, sc, ratio=1.0)
            )

        del lsd
        
    theta_0 = to_half_k(theta_0, args.save_half, args.save_bhalf, vae=vae_key)

    if args.prune:
        theta_0 = prune_model(theta_0, "Model", args, isxl=isxl, isflux=isflux, iszi=iszi)
        
    theta_0 = to_quarter_k(theta_0, args.save_quarter, prefer="e4m3", vae=vae_key)

    for k in tqdm(list(theta_0.keys()), desc="Check contiguous..."):
        if isinstance(theta_0[k], torch.Tensor):
            theta_0[k] = theta_0[k].detach().cpu().contiguous()

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