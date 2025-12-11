from __future__ import annotations
import os
import numpy as np
import json
import argparse
import torch
import torch.nn.functional as F
import scipy.ndimage
import safetensors.torch
import safetensors
from tqdm.auto import tqdm

from Utils import wgt, rand_ratio, sha256, read_metadata_from_safetensors \
    , load_model, parse_ratio, qdtyper, maybe_to_qdtype, diff_inplace \
    , clone_dict_tensors, fineman, weighttoxl, BLOCKID, BLOCKIDFLUX \
    , BLOCKIDXLL, BLOCKIDZI, blockfromkey, checkpoint_dict_skip_on_merge, elementals \
    , to_half, to_half_k, prune_model, cache, merge_cache_json, detect_arch \
    , _swap_components_inplace, _normalize_components_list, _finetune_inplace \
    , _clip_tier_for_xl, _clip_tier_for_flux, _clip_tier_for_zi, _clipxor_semi_hard_blend \
    , _collect_clipxor_targets, _collect_clip_pairs_by_suffix, prepare_merge_cache\
    , trim_delta, normalize_path, prune_extras_vs_model1, unet_permutation_spec \
    , weight_matching, apply_permutation, upcast_fp8_state_dict, to_quarter_k

# Mode Functions

def weight_max(theta0, theta1, *args):
    return torch.max(theta0, theta1)

def geometric(theta0, theta1, alpha):
    return torch.pow(theta0, 1 - alpha) * torch.pow(theta1, alpha)

def sigmoid(theta0, theta1, alpha):
    return (1 / (1 + torch.exp(-4 * alpha))) * (theta0 + theta1) - (1 / (1 + torch.exp(-alpha))) * theta0

def weighted_sum(theta0, theta1, alpha):
    return (1 - alpha) * theta0 + alpha * theta1

def sum_twice(theta0, theta1, theta2, alpha, beta):
    return (1 - beta) * ((1 - alpha) * theta0 + alpha * theta1) + beta * theta2

def triple_sum(theta0, theta1, theta2, alpha, beta):
    return (1 - alpha - beta) * theta0 + alpha * theta1 + beta * theta2

def get_difference(theta1, theta2):
    return theta1 - theta2

def add_difference(theta0, theta1_2_diff, alpha):
    return theta0 + (alpha * theta1_2_diff)

def multiply_difference(theta0, theta1, theta2, alpha, beta):
    theta0_float, theta1_float = theta0.float(), theta1.float()
    diff = (theta0_float - theta2).abs().pow(1 - alpha) * (theta1_float - theta2).abs().pow(alpha)
    sign = weighted_sum(theta0, theta1, beta) - theta2
    return theta2 + torch.copysign(diff, sign).to(theta2.dtype)

def _match_mean_std_like_a(out, a, eps=1e-6):
    a32 = a.detach().float()
    o32 = out.detach().float()
    stdA = a32.std()
    stdO = o32.std()
    if stdO < eps:
        return out
    meanA = a32.mean()
    meanO = o32.mean()
    mean_mix = 0.5 * (meanO + meanA)
    std_mix = 0.5 * (stdO + stdA)
    o32 = (o32 - meanO) / stdO * std_mix + mean_mix
    return o32.to(out.dtype)

def similarity_add_difference(a, b, c, alpha, beta):
    threshold = torch.maximum(a.abs(), b.abs())
    similarity = torch.nan_to_num(((a * b)/(threshold ** 2) + 1) * beta / 2, nan = beta)
    ab_diff = a + alpha * (b - c)
    ab_sum = a * (1 - alpha / 2) + b * (alpha / 2)
    out = torch.lerp(ab_diff, ab_sum, similarity)
    out = _match_mean_std_like_a(out, a)
    return out

def dare_merge(theta0, theta1, alpha, beta):
    if theta0.dim() in (1, 2):
        dw = theta1.shape[-1] - theta0.shape[-1]
        if dw > 0:
            theta0 = F.pad(theta0, (0, dw, 0, 0))
        elif dw < 0: 
            theta1 = F.pad(theta1, (0, -dw, 0, 0))
        dh = theta1.shape[0] - theta0.shape[0]
        if dh > 0:
            theta0 = F.pad(theta0, (0, 0, 0, dh))
        elif dh < 0:
            theta1 = F.pad(theta1, (0, 0, 0, -dh))
    delta = theta1 - theta0
    m = torch.bernoulli(torch.full(delta.shape, float(beta), dtype=torch.float32, device=theta0.device))
    denom = max(1.0 - float(beta), 1e-6)
    delta_hat = (m * delta) / denom
    return theta0 + alpha * delta_hat.to(theta0.dtype)

def feature_weighted_merge(a, b, alpha=0.3, eps=1e-6):
    if a.shape != b.shape or alpha == 0.0:
        return a
    a32 = a.detach().float()
    b32 = b.detach().float()
    delta = trim_delta(b32 - a32, percentile=0.5)
    if delta.dim() == 4 and min(delta.shape[-2], delta.shape[-1]) >= 3:
        delta = F.avg_pool2d(delta, kernel_size=3, stride=1, padding=1)

    stdA = a32.std()
    stdB = b32.std()
    stdDelta = delta.std()
    
    if min(stdA, stdB, stdDelta) < eps:
        return ((1 - alpha) * a32 + alpha * b32).to(a.dtype)
    
    r = (stdB / (stdA + eps)).clamp(0.5, 2.0)
    gamma = 1.0 - 0.5 * (r - 1.0)
    scale = (stdA / (stdDelta + eps)).pow(gamma).clamp(0.5, 1.5)
    tone_corr = (stdA / (stdB + eps)).sqrt().clamp(0.8, 1.1)
    
    merged = a32 + delta * float(alpha) * scale * tone_corr

    meanA = a32.mean()
    stdMerged = merged.std()
    if stdMerged > eps:
        meanMerged = merged.mean()
        mean_mix = 0.5 * (meanMerged + meanA)
        std_mix = 0.5 * (stdMerged + stdA)
        merged = (merged - meanMerged) / stdMerged * std_mix + mean_mix
        
    return merged.to(a.dtype)

def ortho_merge(a, b, alpha):
    a32 = a.detach().float().view(-1)
    d32 = (b.detach().float() - a.detach().float()).view(-1)
    proj = (torch.dot(d32, a32) / (a32.norm()**2 + 1e-12)) * a32
    d_ortho = (d32 - proj).view_as(a)
    return (a + alpha * d_ortho.to(a.dtype)).to(a.dtype)

def sparse_topk(a, b, alpha, beta):
    d = (b.detach().float() - a.detach().float()).abs().view(-1)
    if d.numel() == 0:
        return a
    k = max(int(d.numel() * float(beta)), 1)
    thresh = d.kthvalue(d.numel() - k).values
    mask = ((b.detach().float() - a.detach().float()).abs() >= thresh).to(a.dtype)
    return (a + alpha * (b - a) * mask).to(a.dtype)

def norm_dir_blend(a, b, alpha):
    a32 = a.detach().float().view(-1); b32 = b.detach().float().view(-1)
    an = a32.norm() + 1e-12; bn = b32.norm() + 1e-12
    au = a32 / an; bu = b32 / bn
    du = F.normalize((1 - alpha) * au + alpha * bu, dim=0)
    mag = (1 - alpha) * an + alpha * bn
    out = (du * mag).view_as(a)
    return out.to(a.dtype)

def channel_cosine_gate(a, b, alpha, beta):
    if a.dim() == 4:
        axis = (1,2,3)
    elif a.dim() == 2:
        axis = (1,)
    else:
        return (1 - alpha) * a + alpha * b

    a32 = a.detach().float()
    b32 = b.detach().float()
    num = (a32 * b32).sum(dim=axis)
    den = (a32.norm(dim=axis) * b32.norm(dim=axis) + 1e-12)
    cos = (num / den).clamp_(-1, 1)
    g = (1 - cos) * float(beta)
    while g.dim() < a.dim():
        g = g.unsqueeze(-1)
    mix = (1 - alpha) * a + alpha * b
    return (a * (1 - g) + mix * g).to(a.dtype)

def freq_band_blend(a, b, alpha, beta):
    if a.dim() != 4 or a.shape[-1] < 3 or a.shape[-2] < 3:
        return (1 - alpha) * a + alpha * b
    a32 = a.detach().float(); b32 = b.detach().float()
    A = torch.fft.rfft2(a32, norm="ortho")
    B = torch.fft.rfft2(b32, norm="ortho")
    H, W = a32.shape[-2], a32.shape[-1]
    cut = max(int(min(H, W) * float(beta)), 1)
    yy = torch.arange(A.shape[-2], device=a.device).view(-1,1).float()
    xx = torch.arange(A.shape[-1], device=a.device).view(1,-1).float()
    cy = (A.shape[-2]-1)/2; cx = (A.shape[-1]-1)/2
    dist = torch.sqrt((yy-cy)**2 + (xx-cx)**2)
    low = (dist <= cut).to(A.dtype)
    high = 1 - low
    Aout = low * ((1 - alpha) * A + alpha * A) + high * ((1 - alpha) * A + alpha * B)
    Bout = low * ((1 - alpha) * B + alpha * A) + high * ((1 - alpha) * B + alpha * B)
    F = low * Aout + high * Bout
    out = torch.fft.irfft2(F, s=(H, W), norm="ortho")
    return out.to(a.dtype)

# Mode name assignment

theta_funcs = {
    "WS":   (None,           weighted_sum,               "Weighted Sum"),
    "AD":   (get_difference, add_difference,             "Add Difference"),
    "RM":   (None,           None,                       "Read Metedata"),
    "sAD":  (get_difference, add_difference,             "Smooth Add Difference"),
    "MD":   (None,           multiply_difference,        "Multiply Difference"),
    "SIM":  (None,           similarity_add_difference,  "Similarity Add Difference"),
    "TD":   (None,           add_difference,             "Training Difference"),
    "TS":   (None,           weighted_sum,               "Tensor Sum"),
    "TRS":  (None,           triple_sum,                 "Triple Sum"),
    "ST":   (None,           sum_twice,                  "Sum Twice"),
    "NoIn": (None,           None,                       "No Interpolation"),
    "SIG":  (None,           sigmoid,                    "Sigmoid"),
    "GEO":  (None,           geometric,                  "Geometric"),
    "MAX":  (None,           weight_max,                 "Max"),
    "DARE": (None,           dare_merge,                 "DARE"),
    "XDARE":(None,           dare_merge,                 "CLIP XOR DARE"),
    "ORTHO":(None,           ortho_merge,                "Orthogonalized Delta"),
    "SPRSE":(None,           sparse_topk,                "Sparse Top-k Delta"),
    "NORM": (None,           norm_dir_blend,             "Norm/Direction Split"),
    "CHAN": (None,           channel_cosine_gate,        "Channel-wise Cosine Gate"),
    "FREQ": (None,           freq_band_blend,            "Frequency-Band Blend"),
    "SWAP": (None,           None,                       "Swap Components"),
    "CLIPXOR": (None,        None,                       "CLIP XOR (union-minus-intersection)"),
    "FWM":  (None,           feature_weighted_merge,     "Feature Weighted Merge"),
    "TF":  (None,           None,               "Trim and Fill"),
}
modes_need_m2   = {"sAD", "AD", "TRS", "ST",  "TD", "SIM", "MD", "SPRSE", "HUB", "CHAN", "FREQ"}
modes_need_beta = {"TRS", "ST", "TS",  "SIM", "MD", "DARE"}

parser = argparse.ArgumentParser(description="Merge two or three models")

parser.add_argument("mode",         choices=list(theta_funcs.keys()),   help="Merging mode")
parser.add_argument("model_path",   type=str,                           help="Path to models")
parser.add_argument("model_0",      type=str,                           help="Name of model 0")
parser.add_argument("model_1",      type=str,                           help="Optional, Name of model 1", default=None)
parser.add_argument(f"--model_2",   type=str,                           help="Optional, Name of model 2", default=None, required=False)

for i in range(3):
    parser.add_argument(f"--m{i}_name", type=str, help=f"Custom name of model {i}", default=None, required=False)

for dif in ["10","20","21"]:
    parser.add_argument(f"--use_dif_{dif}", action="store_true", help=f"Use the difference of model {dif[0]} and model {dif[1]} as model {max(int(dif[0]), int(dif[1]))}", required=False)

for p in ["alpha","beta"]:
    parser.add_argument(f"--{p}", default=0.0, help=f"{p.capitalize()} value, optional, defaults to 0", required=False)
    parser.add_argument(f"--rand_{p}", type=str, help=f"Random {p.capitalize()} value, optional", default=None, required=False)

for flag, helpmsg in {
    "cosine0":          "Favor model 0's structure with details from the others (two/three models)",
    "cosine1":          "Favor model 1's structure with details from the others (two/three models)",
    "cosine2":          "Favor model 2's structure with details from the others (three models only)",
    "save_half":        "Save as float16",
    "save_quarter":     "Save as float8",
    "save_bhalf":       "Save as bfloat16",
    "save_safetensors": "Save as .safetensors",
    "keep_ema":         "Keep ema",
    "delete_source":    "Delete the source checkpoint file",
    "no_metadata":      "Save without metadata",
    "prune":            "Prune Model",
    "force":            "Overwrite output if exists",
}.items():
    parser.add_argument(f"--{flag}", action="store_true", help=helpmsg, required=False)

parser.add_argument("--seed",   type=int,   help="Random seed for stochastic modes (e.g., DARE)", default=None)
parser.add_argument("--rebasin",   type=int,   help="ReBasin iterations", default=None)
parser.add_argument("--vae",    type=str,   help="Path of VAE", default=None, required=False)
parser.add_argument("--memo",   type=str,   help="Additional info bake in metadata", default=None)
parser.add_argument("--fine",   type=str,   help="Finetune the given keys on model 0", default=None, required=False)
parser.add_argument("--output",             help="Output file name without extension", default="merged", required=False)
parser.add_argument("--device", type=str,   help="Device to use, defaults to cpu", default="cpu", required=False)

args = parser.parse_args()

if args.save_quarter and args.save_half:
    print("[warn] --save_half and --save_quarter are both set; prioritizing --save_quarter (fp8).")
    args.save_half = False

device = args.device
mode = args.mode
if mode in modes_need_m2 and (args.model_2 is None):
    raise SystemExit(f"mode '{mode}' needs 3rd model")
theta_func1, theta_func2, merge_name = theta_funcs[mode]

if mode not in ["SWAP", "CLIPXOR"]:
    args.alpha, deep_a, block_a = wgt(args.alpha, [])
    args.beta,  deep_b, block_b = wgt(args.beta, [])
    useblocks = block_a or block_b
else:
    useblocks = False
    deep_a = deep_b = []

cos_flags = [args.cosine0, args.cosine1, args.cosine2]
if sum(1 for f in cos_flags if f) > 1:
    raise SystemExit("cosine0, cosine1 and cosine2 cannot be posed at same time, choose one only")

if args.cosine2 and (args.model_2 is None):
    raise SystemExit("--cosine2 cannot be used when there are only 2 models given")

if mode == "WS" and (args.cosine0 ^ args.cosine1):
    cosine0, cosine1 = args.cosine0, args.cosine1
else:
    cosine0 = cosine1 = False
output_name = args.output
output_file = f"{output_name}.{'safetensors' if args.save_safetensors else 'ckpt'}"
output_path = normalize_path(os.path.join(args.model_path, output_file))
merge_cache_json(args.model_path)
cache_data = cache("hashes", None)

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
            output_file = f"{output_name}.{'safetensors' if args.save_safetensors else 'ckpt'}"
            output_path = os.path.join(args.model_path, output_file)
            i += 1
        print(f"Assigned result checkpoint name as {output_file}\n")

stem = lambda p: os.path.splitext(os.path.basename(p))[0]

torch.set_grad_enabled(False)

alpha_seed = beta_seed = None
alpha_info = beta_info = ""
if args.rand_alpha is not None:
    args.alpha, alpha_seed, deep_a, alpha_info = rand_ratio(args.rand_alpha)
if args.rand_beta is not None:
    args.beta, beta_seed,  deep_b,  beta_info  = rand_ratio(args.rand_beta)

model_0_path = normalize_path(os.path.join(args.model_path, args.model_0))
if mode == "RM":
    print(sha256(model_0_path, f"checkpoint/{stem(model_0_path)}"))
    meta = read_metadata_from_safetensors(model_0_path)
    print(json.dumps(meta, indent=2))
    with open(f"./{output_name}.json", "a+", encoding="utf-8") as dmp:
        json.dump(meta, dmp, indent=4)
    exit()
    
interp_method = 2
model_0_name = args.m0_name or stem(model_0_path)
print(f"Loading {model_0_name}...")
theta_0, model_0_sha256, model_0_hash, model_0_meta, cache_data = load_model(model_0_path, device, cache_data=cache_data)
qd0 = qdtyper(theta_0)
isxl, isflux, iszi, theta_0 = detect_arch(theta_0)
theta_0 = upcast_fp8_state_dict(theta_0)

theta_1 = theta_2 = None
model_1_sha256 = model_2_sha256 = None

if mode != "NoIn":
    interp_method = 0
    model_1_path = normalize_path(os.path.join(args.model_path, args.model_1))
    model_1_name = args.m1_name or stem(model_1_path)
    print(f"Loading {model_1_name}...")
    theta_1, model_1_sha256, model_1_hash, model_1_meta, cache_data = load_model(model_1_path, device, cache_data=cache_data)
    qd1 = qdtyper(theta_1)
    theta_1 = upcast_fp8_state_dict(theta_1)
    isxl, isflux, iszi, theta_1 = detect_arch(theta_1)
    if args.fine and not iszi:
        fine = fineman([float(t) for t in args.fine.split(",")], isxl, isflux)
    else:
        fine = ""
        
    if mode == "SWAP":
        components = _normalize_components_list(str(args.alpha))
        if not components:
            components = {"unet", "vae", "clip-l", "clip-g", "clip", "transformer", "text", "text2"}

        moved, created, skipped, theta_0 = _swap_components_inplace(theta_0, theta_1, components, isxl, isflux, iszi)
        print(f"[SWAP] components={sorted(list(components))}  moved:{moved}  created:{created}  shape_skipped:{skipped}")
        
        mode = "NoIn"
        theta_1 = None
        usebeta = False
        weights_a = weights_b = None
        alpha = beta = None
        
    elif mode in ["CLIPXOR", "XDARE"]:
        theta_res = clone_dict_tensors(theta_0)
        
        base_hardness = 0.70
        hard_l = base_hardness
        hard_g = base_hardness

        hard_t5   = 0.60
        hard_clip = base_hardness
        
        isxl_a, isflux_a, iszi_a, theta_0 = detect_arch(theta_0)
        isxl_b, isflux_b, iszi_b, theta_1 = detect_arch(theta_1)

        targets = _collect_clipxor_targets(theta_0, theta_1, isxl=isxl_a, isflux=isflux_a, iszi=iszi_a)
        if not targets:
            suffix_pairs = _collect_clip_pairs_by_suffix(theta_0, theta_1, isxl_a, isflux_a, iszi_a, isxl_b, isflux_b, iszi_b)
            targets = [ka for (_, ka, _) in suffix_pairs]
            
        if not targets:
            print("[CLIPXOR] No eligible CLIP keys to merge (even after suffix matching). \nArchitectures may be incompatible or shapes differ.")
        else:
            for key_a in tqdm(targets, desc="CLIPXOR: Collecting keys...", total=len(targets)):
                A = theta_0[key_a]
                if key_a in theta_1:
                    key_b = key_a
                else:
                    if not suffix_pairs:
                        continue
                    pass
        suffix_to_kb = {}
        if suffix_pairs:
            for suf, ka, kb in suffix_pairs:
                suffix_to_kb[ka] = kb
        
        for key in tqdm(targets, desc="CLIPXOR merging...", total=len(targets)):
            A = theta_0[key_a]
            key_b = key_a if key_a in theta_1 else suffix_to_kb.get(key_a, None)
            if key_b is None:
                continue
            B = theta_1[key_b]

            if isxl_a or isxl_b:
                tier = _clip_tier_for_xl(key_a)
                hardness = hard_l if tier == "clip-l" else (hard_g if tier == "clip-g" else base_hardness)
            elif isflux_a or isflux_b:
                tier = _clip_tier_for_flux(key_a)
                hardness = hard_t5 if tier == "t5" else (hard_clip if tier == "clip" else base_hardness)
            elif iszi_a or iszi_b:
                tier = _clip_tier_for_zi(key_a)
                hardness = hard_t5 if tier == "qwen3_4b" else (hard_clip if tier == "cap_embedder" else base_hardness)
            else:
                hardness = base_hardness

            M_semi = _clipxor_semi_hard_blend(
                A, B,
                hardness=float(hardness),
                use_cosine_gate=True,
                keep_stats=True
            )
            if 'fine' in locals() and fine:
                M_semi = _finetune_inplace(key_a, M_semi, fine)
            theta_res[key_a] = M_semi

        theta_0 = theta_res
        if mode == "CLIPXOR":
            mode = "NoIn"
            theta_1 = None
            usebeta = False
            weights_a = weights_b = None
            alpha = beta = None
        elif mode == "XDARE":
            mode = "DARE"
            usebeta = True
            weights_a, alpha, alpha_info = parse_ratio(args.alpha, alpha_info, deep_a)
            weights_b, beta, beta_info = parse_ratio(args.beta, beta_info, deep_b)
    else:
        weights_a, alpha, alpha_info = parse_ratio(args.alpha, alpha_info, deep_a)
        if mode in modes_need_m2:
            model_2_path = normalize_path(os.path.join(args.model_path, args.model_2))
            model_2_name = args.m2_name or stem(model_2_path)
            print(f"Loading {model_2_name}...")
            theta_2, model_2_sha256, model_2_hash, model_2_meta, cache_data = load_model(model_2_path, device, cache_data=cache_data)
            qd2 = qdtyper(theta_2)
            isxl, isflux, iszi, theta_2 = detect_arch(theta_2)
            theta_2 = upcast_fp8_state_dict(theta_2)

        usebeta = mode in modes_need_beta
        if usebeta:
            weights_b, beta, beta_info = parse_ratio(args.beta, beta_info, deep_b)
        else:
            weights_b, beta = None, None
        if args.rebasin is not None:
            if isflux or iszi:
                print("[ReBasin] Unavailable architecture detected, skipping ReBasin (not supported).")
            else:
                print(f"[ReBasin] Running weight matching (Hungarian)... iter={args.rebasin}")
                ps = unet_permutation_spec(isxl)
                perm_01, gain_01 = weight_matching(
                    ps,
                    params_a=theta_0,
                    params_b=theta_1,
                    max_iter=args.rebasin,
                    usefp16=True,
                    device=device,
                )
                theta_1 = apply_permutation(ps, perm_01, theta_1)
                print(f"[ReBasin] (0 <-> 1) average gain: {gain_01:.4f}")

                if mode in modes_need_m2 and theta_2 is not None:
                    perm_02, gain_02 = weight_matching(
                        ps,
                        params_a=theta_0,
                        params_b=theta_2,
                        max_iter=args.rebasin,
                        usefp16=True,
                        device=device,
                    )
                    theta_2 = apply_permutation(ps, perm_02, theta_2)
                    print(f"[ReBasin] (0 <-> 2) average gain: {gain_02:.4f}")

else:
    usebeta = False
    weights_a = weights_b = None
    alpha = beta = None
    isxl, isflux, iszi = False, False, False

if args.vae:
    vae_name = stem(args.vae)
    vae, *_ = load_model(normalize_path(args.vae), device, verify_hash=False)

if mode == "DARE":
    g = torch.Generator(device=device if device != "cpu" else "cpu")
    if args.seed is not None: g.manual_seed(args.seed)

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

def cosine_minmax_grouped(base_dict, other_dict, desc, variant=0, lo=10.0, hi=90.0):
    by_block = {}
    for k in tqdm(base_dict.keys(), desc=desc):
        if "first_stage_model" in k or ("model" not in k and "text_encoders" not in k) or k not in other_dict:
            continue
        wi = _resolve_weight_index(k)
        if wi < 0:
            continue
        a = base_dict[k].detach().float().view(-1)
        b = other_dict[k].detach().float().view(-1)
        if a.numel() == 0 or b.numel() == 0 or a.shape != b.shape:
            continue
        if _is_small_or_norm_or_bias(k, base_dict[k]):
            continue

        cos = F.cosine_similarity(a, b, dim=0)
        cos = torch.nan_to_num(cos, nan=0.0, posinf=1.0, neginf=-1.0)

        if variant == 1:
            dot = torch.dot(a, b)
            denom = float(a.norm() * b.norm()) + 1e-12
            mag = (dot / denom)
            sim = 0.5 * (cos + mag)
        else:
            sim = cos

        sim = float(torch.clamp(sim, -1.0, 1.0))
        by_block.setdefault(wi, []).append(sim)

    stats = {}
    for wi, vals in by_block.items():
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            stats[wi] = (0.0, 1.0)
            continue
        lo_v = float(np.percentile(arr, lo))
        hi_v = float(np.percentile(arr, hi))
        if hi_v - lo_v < 1e-6:
            hi_v = lo_v + 1e-6
        stats[wi] = (max(-1.0, lo_v), min(1.0, hi_v))
    return stats, (0.0, 1.0)

if theta_func1:
    if isflux:
        theta_1, theta_2 = maybe_to_qdtype(theta_1, theta_2, qd1, qd2, device)
    diff_inplace(theta_1, theta_2, theta_func1, "Getting Difference of Model 1 and 2")
    del theta_2

if isflux:
    theta_0, theta_1 = maybe_to_qdtype(theta_0, theta_1, qd0, qd1, device)
    if 'theta_2' in locals() and theta_2 is not None:
        theta_0, theta_2 = maybe_to_qdtype(theta_0, theta_2, qd0, qd2, device)

# if mode == "TS":
#     theta_0 = clone_dict_tensors(theta_0)
    
if args.use_dif_21:
    # theta_2 := model1 - model2
    diff_inplace(theta_2, theta_1, get_difference, "Getting Difference of Model 1 and 2")

if args.use_dif_10:
    # theta_1 := model1 - model0
    diff_inplace(theta_1, theta_0, get_difference, "Getting Difference of Model 0 and 1")

if args.use_dif_20:
    # theta_2 := model2 - model0
    diff_inplace(theta_2, theta_0, get_difference, "Getting Difference of Model 0 and 2")

def resolve_cosine_triplet(theta_0, theta_1, theta_2, use_cos0, use_cos1, use_cos2):
    if use_cos0:
        base, dA, dB = theta_0, theta_1, theta_2
        varA, varB = 0, 0
    elif use_cos1:
        base, dA, dB = theta_1, theta_0, theta_2
        varA, varB = 1, 0
    else:
        base, dA, dB = theta_2, theta_0, theta_1
        varA, varB = 0, 0
    return base, dA, dB, varA, varB

if mode not in ["NoIn", "TF"]:
    if isxl and useblocks:
        print("Detected XL architecture.")
        if len(weights_a) == 25:
            weights_a = weighttoxl(weights_a)
            print(f"alpha weight converted for XL{weights_a}")
        elif len(weights_a) == 19:
            weights_a += [0]
        if mode in modes_need_m2 and usebeta:
            if len(weights_b) == 25:
                weights_b = weighttoxl(weights_b)
                print(f"beta weight converted for XL{weights_b}")
            elif len(weights_b) == 19:
                weights_b += [0]
    elif iszi and useblocks:
        print("Detected Zimage architecture.")
        if len(weights_a) > 34:
            weights_a = weights_a[:34]
            print(f"alpha weight converted for Zimage{weights_a}")
        elif len(weights_a) < 34:
            weights_a += [0] * (34 - len(weights_a))
        if mode in modes_need_m2 and usebeta:
            if len(weights_b) > 34:
                weights_b = weights_b[:34]
                print(f"beta weight converted for Zimage{weights_b}")
            elif len(weights_b) < 34:
                weights_b += [0] * (34 - len(weights_b))
        
def _resolve_weight_index(key):
    block, tag = blockfromkey(key, isxl, isflux, iszi)
    if block == "Not Merge":
        return -1
    if isflux and tag in BLOCKIDFLUX: return BLOCKIDFLUX.index(tag)
    if isxl   and tag in BLOCKIDXLL:  return BLOCKIDXLL.index(tag)
    if iszi   and tag in BLOCKIDZI:    return BLOCKIDZI.index(tag)
    if tag in BLOCKID:                return BLOCKID.index(tag)
    return -1

def _apply_cosine_blend(a, b, kmin, kmax, cur_alpha, variant, tau=0.20, floor=0.05):
    a_f = a.detach().float().view(-1)
    b_f = b.detach().float().view(-1)

    sim = F.cosine_similarity(a_f, b_f, dim=0)
    sim = torch.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=-1.0)

    if variant == 1:
        dot = torch.dot(a_f, b_f)
        denom = (a_f.norm() * b_f.norm()).clamp_min(1e-12)
        mag = (dot / denom).clamp_(-1.0, 1.0)
        sim = 0.5 * (sim + mag)

    sim = sim.clamp_(-1.0, 1.0)
    kmin = float(np.clip(kmin, -1.0, 1.0))
    kmax = float(np.clip(kmax, -1.0, 1.0))
    if abs(kmax - kmin) < 1e-6:
        kmax = kmin + 1e-6
        
    t = ((sim - kmin) / (kmax - kmin)).clamp_(0.0, 1.0)
    mid = 0.5 + float(cur_alpha) * 0.5
    w = torch.sigmoid((t - mid) / max(tau, 1e-3))
    w = w.clamp(floor, 1.0 - floor)
    
    out = torch.lerp(a, b, w).view_as(a).to(a.dtype)
    return out

use_cos0 = bool(args.cosine0)
use_cos1 = bool(args.cosine1)
use_cos2 = bool(args.cosine2)

if use_cos0 or use_cos1 or use_cos2:
    base, dA, dB, varA, varB = resolve_cosine_triplet(theta_0, theta_1, theta_2, use_cos0, use_cos1, use_cos2)

    statsA, defaultA = cosine_minmax_grouped(base, dA, "Cosine(base vs A)", variant=varA)
    if dB is not None:
        statsB, defaultB = cosine_minmax_grouped(base, dB, "Cosine(base vs B)", variant=varB)
    else:
        statsB = {}; defaultB = (0.0, 1.0)

    # theta_res = clone_dict_tensors(base)

    for key in tqdm(base.keys(), desc="Cosine structure-based blending..."):
        if "first_stage_model" in key or ("model" not in key and "text_encoders" not in key):
            continue
        if key not in dA:
            continue

        wi = _resolve_weight_index(key)
        if wi < 0:
            continue

        if _is_small_or_norm_or_bias(key, base[key]):
            cur_a = alpha
            if weights_a is not None and wi > 0: cur_a = weights_a[wi - 1]
            if deep_a: cur_a = elementals(key, wi, deep_a, cur_a)
            out = weighted_sum(base[key], dA[key], cur_a)
            if dB is not None and (key in dB) and (beta is not None):
                cur_b = beta
                if weights_b is not None and wi > 0: cur_b = weights_b[wi - 1]
                if deep_b: cur_b = elementals(key, wi, deep_b, cur_b)
                out = weighted_sum(out, dB[key], cur_b)
            base[key] = _finetune_inplace(key, out, fine)
            continue

        cur_a, cur_b = alpha, beta
        if wi > 0:
            if weights_a is not None:            cur_a = weights_a[wi - 1]
            if (weights_b is not None) and dB is not None: cur_b = weights_b[wi - 1]
        if deep_a: cur_a = elementals(key, wi, deep_a, cur_a)
        if deep_b and dB is not None: cur_b = elementals(key, wi, deep_b, cur_b)

        ka = statsA.get(wi, defaultA); kminA, kmaxA = ka
        out = _apply_cosine_blend(base[key], dA[key], kminA, kmaxA, cur_a, variant=varA, tau=0.20, floor=0.05)

        if dB is not None and (key in dB) and (cur_b is not None):
            kb = statsB.get(wi, defaultB); kminB, kmaxB = kb
            out = _apply_cosine_blend(out, dB[key], kminB, kmaxB, cur_b, variant=varB, tau=0.20, floor=0.05)

        base[key] = _finetune_inplace(key, out, fine)

    theta_0 = base

def remerge_model(target_dict, source_dict, desc, mode, theta_2=None):
    for key in tqdm(source_dict.keys(), desc=desc):
        if isflux or key in checkpoint_dict_skip_on_merge or ("model" not in key and "text_encoders" not in key) or key in target_dict:
            continue

        cache_entry = merge_cache.get(key)
        if cache_entry is None:
            target_dict[key] = source_dict[key]
            continue

        _,_,cur_b = cache_entry

        if mode in {"TRS", "ST"} and theta_2 is not None and key in theta_2:
            b, c = source_dict[key], theta_2[key]
            try:
                target_dict[key] = weighted_sum(b, c, cur_b)
            except Exception:
                target_dict[key] = b
        else:
            target_dict[key] = source_dict[key]

    return target_dict

if mode not in ["NoIn", "TF"]:
    merge_cache = prepare_merge_cache(theta_0.keys(), isxl, isflux, iszi, deep_a, deep_b, weights_a, weights_b, alpha, beta)
    vae_key = "first_stage_model" if not (isflux or iszi) else "vae"
    
    for key in tqdm(theta_0.keys(), desc=f"{merge_name} Merging..."):
        if args.vae is None and vae_key in key:
            continue
        if not (theta_1 and ("model" in key or "text_encoders" in key) and key in theta_1):
            continue
        if mode != "DARE" and (usebeta or mode == "TD") and (theta_2 is not None) and key not in theta_2:
            continue
        if key in checkpoint_dict_skip_on_merge:
            continue
        if key not in merge_cache: 
            continue

        a, b = theta_0[key], theta_1[key]
        al, bl = list(a.shape), list(b.shape)

        wi, cur_a, cur_b = merge_cache[key]
        a, b = theta_0[key], theta_1[key]

        if mode == "sAD":
            bf = b.detach().float()
            filt = scipy.ndimage.gaussian_filter(bf.cpu().numpy(), sigma=1)
            theta_0[key] = a + cur_a * torch.from_numpy(filt).to(a.device, dtype=a.dtype)
            theta_0[key] = _finetune_inplace(key, theta_0[key], fine); continue

        if mode == "TD":
            t1 = theta_1[key].float()
            t2 = theta_2[key].float()
            t0 = theta_0[key].float()
            if torch.allclose(t1, t2, rtol=0, atol=0):
                theta_2[key] = theta_0[key]
                continue
            diff_AB = (t1 - t2).abs()
            dist_A0 = (t1 - t0).abs()
            dist_A2 = (t1 - t2).abs()
            denom = dist_A0 + dist_A2
            scale = torch.where(denom != 0, dist_A0 / denom, torch.tensor(0., device=t0.device))
            scale = torch.sign(t1 - t2) * scale.abs()
            theta_0[key] = (t0 + (scale * diff_AB) * (cur_a * 1.8)).to(theta_0[key].dtype)
            theta_0[key] = _finetune_inplace(key, theta_0[key], fine); continue

        if mode == "TS":
            if a.dim() == 0:
                continue
            n = a.shape[0]
            if cur_a + cur_b <= 1:
                s, e = int(n * cur_b), int(n * (cur_a + cur_b))
                theta_0[key][s:e, ...] = b[s:e, ...].clone()
            else:
                s, e = int(n * (cur_a + cur_b - 1)), int(n * cur_b)
                t = b.clone()
                t[s:e, ...] = a[s:e, ...].clone()
                theta_0[key] = t
            theta_0[key] = _finetune_inplace(key, theta_0[key], fine)
            continue

        if al != bl and len(al) == 4 and len(bl) == 4 and al[0]==bl[0] and al[2:]==bl[2:]:
            use = min(al[1], bl[1], 4)
            ad = a[:, :use, ...]
        else:
            ad = a

        if usebeta and mode not in ["DARE", "XDARE"]:
            c = theta_2[key]
            theta_0[key] = theta_func2(ad, b, c, cur_a, cur_b)
        elif usebeta:
            theta_0[key] = theta_func2(ad, b, cur_a, cur_b)
        else:
            theta_0[key] = theta_func2(ad, b, cur_a)

        theta_0[key] = _finetune_inplace(key, theta_0[key], fine)
        
    merge_cache = prepare_merge_cache(
        list(set(theta_1.keys()) | (set(theta_2.keys()) if 'theta_2' in locals() and theta_2 else set())),
        isxl, isflux, iszi, deep_a, deep_b, weights_a, weights_b, alpha, beta
    )

    if mode != "DARE":
        if mode != "AD":
            theta_0 = remerge_model(theta_0, theta_1, desc="Remerging...", mode=mode, theta_2=theta_2)
        else:
            theta_0 = remerge_model(theta_0, theta_1, desc="Remerging...", mode=mode)
    del theta_1
    try:
        if theta_2:
            theta_0 = remerge_model(theta_0, theta_2, desc="Remerging...", mode=mode)
            del theta_2
    except NameError:
        pass

else:
    if args.mode == "TF":
        theta_0 = prune_extras_vs_model1(theta_0, theta_1)
        theta_0 = remerge_model(theta_0, theta_1, desc="Remerging...", mode=mode, theta_2=theta_2)
    isxl, isflux, iszi, theta_0 = detect_arch(theta_0)
    vae_key = "first_stage_model" if not (isflux or iszi) else "vae"
    if args.fine and not iszi:
        fine = fineman([float(t) for t in args.fine.split(",")], isxl, isflux)
        for key in tqdm(theta_0.keys(), desc="Fine Tuning ..."):
            if args.vae is None and vae_key in key:
                continue
            theta_0[key] = _finetune_inplace(key, theta_0[key], fine)
    else:
        fine = ""
        
if args.vae:
    for k in tqdm(vae.keys(), desc=f"Baking in VAE[{vae_name}] ..."):
        tk = vae_key + "." + k
        theta_0[tk] = to_half(vae[k], args.save_half)
    del vae

isxl, isflux, iszi, theta_0 = detect_arch(theta_0)

if isxl:
    for k in tqdm([k for k in theta_0.keys() if "cond_stage_model." in k], desc="Cond resolving..."):
        del theta_0[k]

theta_0 = to_half_k(theta_0, args.save_half, args.save_bhalf, vae=vae_key)

if args.prune:
    theta_0 = prune_model(theta_0, "Model", args, isxl, isflux, iszi)

theta_0 = to_quarter_k(theta_0, args.save_quarter, prefer="e4m3", vae=vae_key)

for k in tqdm(theta_0.keys(), desc="Check contiguous..."):
    theta_0[k] = theta_0[k].contiguous()

metadata = {"format": "safetensors" if args.save_safetensors else "ckpt", "sd_merge_models": {}, "sd_merge_recipe": None}
if args.memo is not None:
    metadata["memo"] = args.memo

calcs = [
    name for flag, name in [
        (cosine0,            "cosine_0"),
        (cosine1,            "cosine_1"),
        (args.use_dif_10,    "use_dif_10"),
        (args.use_dif_20,    "use_dif_20"),
        (args.use_dif_21,    "use_dif_21"),
    ] if flag
]
if args.fine:
    calcs.append(f"fine[{fine}]")
calcl = ",".join(calcs) or None

fp = "fp8" if args.save_quarter else ("fp16" if args.save_half else "fp32")

merge_recipe = {
    "type":                 "merge-models-chattiori",
    "primary_model_hash":   model_0_sha256,
    "secondary_model_hash": model_1_sha256 if mode != "NoIn" else None,
    "tertiary_model_hash":  model_2_sha256 if mode in modes_need_m2 else None,
    "merge_method":         merge_name,
    "block_weights":        (weights_a is not None or weights_b is not None),
    "alpha_info":           alpha_info or None,
    "beta_info":            beta_info  or None,
    "calculation":          calcl,
    "fp":                   fp,
    "output_name":          output_name,
    "bake_in_vae":          (vae_name if args.vae else False),
    "pruned":               args.prune,
}

if args.mode == "SWAP":
    merge_recipe["swap_components_alpha_text"] = str(args.alpha)
elif args.mode == "CLIPXOR":
    merge_recipe["clipxor"] = {"intersection": "elemwise_minabs_same_sign", "base": False}
elif args.rebasin is not None:
    merge_recipe["rebasin"] = {
        "iter": args.rebasin,
        "min_channels": 64,
        "max_channels": 4096,
    }
    
metadata["sd_merge_recipe"] = json.dumps(merge_recipe)

def add_model_metadata(s256, hashed, meta, model_name):
    metadata["sd_merge_models"][s256] = {
        "name": model_name,
        "legacy_hash": hashed,
        "sd_merge_recipe": meta.get("sd_merge_recipe"),
    }
    metadata["sd_merge_models"].update(meta.get("sd_merge_models", {}))

add_model_metadata(model_0_sha256, model_0_hash, model_0_meta, model_0_name)
if mode != "NoIn":
    add_model_metadata(model_1_sha256, model_1_hash, model_1_meta, model_1_name)
if mode in modes_need_m2:
    add_model_metadata(model_2_sha256, model_2_hash, model_2_meta, model_2_name)

metadata["sd_merge_models"] = json.dumps(metadata["sd_merge_models"])

delete_targets = []
if args.delete_source:
    for p, cond in [
        (os.path.join(args.model_path, args.model_0), True),
        (os.path.join(args.model_path, args.model_1), mode != "NoIn"),
        (os.path.join(args.model_path, args.model_2), mode in modes_need_m2),
    ]:
        if cond and os.path.isfile(p):
            delete_targets.append(p)

merge_success = False
try:
    print(f"Saving as {output_file}...")
    if args.save_safetensors:
        with torch.no_grad():
            safetensors.torch.save_file(
                theta_0, output_path,
                metadata=None if args.no_metadata else metadata
            )
    else:
        torch.save({"state_dict": theta_0}, output_path)

    merge_success = True
    print(f"Done! ({round(os.path.getsize(output_path)/1073741824, 2)}G)")

finally:
    if args.delete_source and merge_success:
        for p in delete_targets:
            try:
                os.remove(p)
                print(f"[delete_source] Removed source: {p}")
            except Exception as e:
                print(f"[delete_source] Failed to remove {p}: {e}")