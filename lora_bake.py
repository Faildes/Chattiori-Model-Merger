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
    BLOCKID,
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

def merge_weights(lora: dict, isv2: bool, isxl: bool, p: float, lam: float, scale: float, strengths: list[float]):
    out = {}
    for k, v in lora.items():
        full = convert_diffusers_name_to_compvis(k, isv2)
        msd  = full.split(".", 1)[0]
        if isxl:
            msd = msd.replace("lora_unet", "diffusion_model").replace("lora_te1_text_model", "0_transformer_text_model")
        strength = strengths[0]
        for i, b in enumerate(LBLOCKS26):
            if b in full or b in msd:
                strength = strengths[i] if i < len(strengths) else strengths[0]
                break
        out[k] = strength * lam * apply_dare(v, p)
    return apply_spectral_norm(out, scale) if scale > 0 else out

def get_loralist(arg: str):
    return [x.split(":", 1) if ":" in x else [x, "1.0"] for x in arg.split(",") if x.strip()]

def _build_keymap(sd: dict):
    km = {}
    for k in sd.keys():
        if "model" not in k: 
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
    vae_key = "first_stage_model" if not (isflux or iszi) else "vae"

    keymap   = _build_keymap(theta_0)
    lr_strs  = []
    lora_meta = {}

    for lora_model, ratio_str in lora_list:
        print(f"loading: {lora_model}")
        ratios = ([float(x) for x in ratio_str.replace(" ", "").split(",")] 
                  if isinstance(ratio_str, str) else [ratio_str] * len(BLOCKID))
        lr_strs.append("[" + ",".join(str(x) for x in ratios) + "]")

        lpath = normalize_path(os.path.join(model_path, lora_model))
        lsd, meta, lisv2 = load_state_dict(lpath, torch.float)
        lhash, _, cache_data = sha256_from_cache(lpath, f"lora/{os.path.splitext(os.path.basename(lpath))[0]}", cache_data)
        lora_meta[lhash] = meta

        for k in tqdm(list(lsd.keys()), desc=f"Merging {lora_model}..."):
            if "lora_down" not in k: 
                continue
            up_k    = k.replace("lora_down", "lora_up")
            alpha_k = k[:k.index("lora_down")] + "alpha"

            full = convert_diffusers_name_to_compvis(k, lisv2)
            msd  = full.split(".", 1)[0]
            if isxl:
                msd = msd.replace("lora_unet","diffusion_model").replace("lora_te1_text_model","0_transformer_text_model")
            if msd not in keymap: 
                continue

            ratio = ratios[0]
            for i, b in enumerate(LBLOCKS26):
                if b in full or b in msd:
                    ratio = ratios[i] if i < len(ratios) else ratios[0]
                    break

            W     = theta_0[keymap[msd]].to("cpu")
            down  = lsd[k].to("cpu")
            up    = lsd[up_k].to("cpu")
            dim   = down.size(0)
            alpha = lsd.get(alpha_k, dim)
            sc    = (alpha / dim)

            theta_0[keymap[msd]] = torch.nn.Parameter(_apply_lora_to_weight(W, up, down, sc, ratio))

        del lsd
        
    theta_0 = to_half_k(theta_0, args.save_half, args.save_bhalf, vae=vae_key)

    if args.prune:
        theta_0 = prune_model(theta_0, "Model", args, isxl=isxl, isflux=isflux, iszi=iszi)

    theta_0 = to_quarter_k(theta_0, args.save_quarter, prefer="e4m3", vae=vae_key)

    for k in tqdm(list(theta_0.keys()), desc="Check contiguous..."):
        theta_0[k] = theta_0[k].contiguous()

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
    vae_key = "first_stage_model" if not (isflux or iszi) else "vae"
    keymap = _build_keymap(theta_0)

    main_sd, _, mlv2 = load_state_dict(mainlora, torch.float, depatch=False)

    lam, p, scale = 1.5, 0.13, 0.2
    torch.manual_seed(0)

    lr_strs, lora_meta = [], {}

    for lora_model, ratio_str in lora_list:
        print(f"loading: {lora_model}")
        ratios = ([float(x) for x in ratio_str.replace(" ", "").split(",")] 
                  if isinstance(ratio_str, str) else [ratio_str] * len(BLOCKID))
        lr_strs.append("[" + ",".join(str(x) for x in ratios) + "]")

        lpath = normalize_path(os.path.join(model_path, lora_model))
        lsd, meta, lisv2 = load_state_dict(lpath, torch.float, depatch=False)
        lhash, _, cache_data = sha256_from_cache(lpath, f"lora/{os.path.splitext(os.path.basename(lpath))[0]}", cache_data)
        lora_meta[lhash] = meta

        lw = merge_weights(lsd, lisv2, isxl, p, lam, scale, ratios)

        for k in tqdm(list(main_sd.keys()), desc=f"Merging {lora_model}..."):
            if "lora_down" not in k or k not in lw: 
                continue
            full = convert_diffusers_name_to_compvis(k, mlv2)
            msd  = full.split(".", 1)[0]
            if isxl:
                msd = msd.replace("lora_unet","diffusion_model").replace("lora_te1_text_model","0_transformer_text_model")
            if msd not in keymap:
                continue

            up_k   = k.replace("lora_down", "lora_up")
            alpha_k = k[:k.index("lora_down")] + "alpha"

            down, up = lw[k].to("cpu"), lw[up_k].to("cpu")
            dim = down.size(0)
            sc  = lw.get(alpha_k, dim) / dim

            W = theta_0[keymap[msd]].to("cpu")
            theta_0[keymap[msd]] = torch.nn.Parameter(_apply_lora_to_weight(W, up, down, sc, ratio=1.0))

        del lsd
        
    theta_0 = to_half_k(theta_0, args.save_half, args.save_bhalf, vae=vae_key)

    if args.prune:
        theta_0 = prune_model(theta_0, "Model", args, isxl=isxl, isflux=isflux, iszi=iszi)
        
    theta_0 = to_quarter_k(theta_0, args.save_quarter, prefer="e4m3", vae=vae_key)

    for k in tqdm(list(theta_0.keys()), desc="Check contiguous..."):
        theta_0[k] = theta_0[k].contiguous()

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