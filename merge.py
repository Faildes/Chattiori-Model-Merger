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
from collections import OrderedDict

from Utils import wgt, rand_ratio, sha256, read_metadata_from_safetensors \
    , load_model, parse_ratio, qdtyper, maybe_to_qdtype, diff_inplace \
    , fineman, weighttoxl, BLOCKID, BLOCKIDFLUX, BLOCKIDXLL, BLOCKIDZI, BLOCKIDAM \
    , blockfromkey, checkpoint_dict_skip_on_merge, elementals2, extra_tag_for_key, _is_small_or_norm_or_bias \
    , to_half, cache, set_cache_filename, base_path, merge_cache_json, detect_arch \
    , _swap_components_inplace, _normalize_components_list, _finetune_inplace \
    , _clip_tier_for_xl, _clip_tier_for_flux, _clip_tier_for_zi, _clipxor_semi_hard_blend \
    , _collect_clipxor_targets, _collect_clip_pairs_by_suffix, turbo_convert_inplace \
    , trim_delta, normalize_path, prune_extras_vs_model1, unet_permutation_spec \
    , weight_matching, apply_permutation, upcast_fp8_state_dict, _parse_components_with_only \
    , _common_dtype, _filter_state_dict_by_components, prepare_state_dict_for_save, apply_vae_saturation_inplace, normalize_external_text_encoder

# Mode Functions

def weight_max(theta0, theta1, *args):
    return torch.maximum(theta0, theta1)

def geometric(theta0, theta1, alpha):
    return torch.pow(theta0, 1 - alpha) * torch.pow(theta1, alpha)

def sigmoid(theta0, theta1, alpha):
    a = float(alpha)
    s1 = 1.0 / (1.0 + np.exp(-4.0 * a))
    s0 = 1.0 / (1.0 + np.exp(-1.0 * a))
    return (s1 * (theta0 + theta1) - s0 * theta0)

def weighted_sum(theta0, theta1, alpha):
    if theta1.dtype != theta0.dtype: theta1 = theta1.to(theta0.dtype)
    return torch.lerp(theta0, theta1, float(alpha))

@torch.inference_mode()
def sum_twice(theta0, theta1, theta2, alpha, beta):
    dt = _common_dtype(theta0, theta1, theta2)
    if theta0.dtype != dt: theta0 = theta0.to(dt)
    if theta1.dtype != dt: theta1 = theta1.to(dt)
    if theta2.dtype != dt: theta2 = theta2.to(dt)

    out = torch.empty_like(theta0)
    torch.lerp(theta0, theta1, float(alpha), out=out)     # out = lerp(a,b,alpha)
    torch.lerp(out,   theta2, float(beta),  out=out)      # out = lerp(out,c,beta)
    return out

@torch.inference_mode()
def triple_sum(theta0, theta1, theta2, alpha, beta):
    dt = _common_dtype(theta0, theta1, theta2)
    if theta0.dtype != dt: theta0 = theta0.to(dt)
    if theta1.dtype != dt: theta1 = theta1.to(dt)
    if theta2.dtype != dt: theta2 = theta2.to(dt)

    a = float(alpha); b = float(beta)
    c0 = 1.0 - a - b

    out = torch.empty_like(theta0)
    torch.mul(theta0, c0, out=out)         # out = (1-a-b)*theta0
    out.add_(theta1, alpha=a)              # out += a*theta1
    out.add_(theta2, alpha=b)              # out += b*theta2
    return out

def get_difference(theta1, theta2):
    return theta1 - theta2

def add_difference(theta0, theta1_2_diff, alpha):
    return theta0 + (alpha * theta1_2_diff)

def multiply_difference(theta0, theta1, theta2, alpha, beta):
    a = theta0.float()
    b = theta1.float()
    c = theta2.float() if theta2.dtype != torch.float32 else theta2

    # diff = |a-c|^(1-a) * |b-c|^a
    da = (a - c).abs()
    db = (b - c).abs()
    diff = da.pow(1.0 - float(alpha)).mul_(db.pow(float(alpha)))

    # sign = ((1-beta)*theta0 + beta*theta1) - theta2
    sign = torch.lerp(theta0, theta1, float(beta)).float().sub_(c)

    out = c + torch.where(sign >= 0, diff, -diff)
    return out.to(theta2.dtype)


_SIM_SCRATCH = OrderedDict()
_SIM_MAX_SCRATCH = 64

def _sim_buf(device, dtype, shape):
    key = (str(device), dtype, tuple(shape))
    buf = _SIM_SCRATCH.get(key)
    if buf is not None:
        _SIM_SCRATCH.move_to_end(key)
        return buf
    thr = torch.empty(shape, device=device, dtype=dtype)
    sim = torch.empty(shape, device=device, dtype=dtype)
    out = torch.empty(shape, device=device, dtype=dtype)
    _SIM_SCRATCH[key] = (thr, sim, out)
    if len(_SIM_SCRATCH) > _SIM_MAX_SCRATCH:
        _SIM_SCRATCH.popitem(last=False)
    return thr, sim, out

def clear_sim_scratch():
    _SIM_SCRATCH.clear()
    

def _match_mean_std_like_a(out, a, eps=1e-6):
    a32 = a if a.dtype == torch.float32 else a.detach().float()
    o32 = out if out.dtype == torch.float32 else out.detach().float()

    varA, meanA = torch.var_mean(a32, unbiased=False)
    varO, meanO = torch.var_mean(o32, unbiased=False)

    stdA = varA.sqrt()
    stdO = varO.sqrt()
    if stdO < eps:
        return out

    mean_mix = 0.5 * (meanO + meanA)
    std_mix  = 0.5 * (stdO + stdA)
    o32 = (o32 - meanO) / stdO * std_mix + mean_mix
    return o32.to(out.dtype)

@torch.inference_mode()
def similarity_add_difference(a, b, c, alpha, beta):
    a_orig_dtype = a.dtype
    dev = a.device
    if b.device != dev: b = b.to(dev)
    if c.device != dev: c = c.to(dev)

    dt = _common_dtype(a, b, c)
    if a.dtype != dt: a = a.to(dt)
    if b.dtype != dt: b = b.to(dt)
    if c.dtype != dt: c = c.to(dt)

    a2 = float(alpha) * 0.5
    b2 = float(beta)  * 0.5

    thr, sim, out = _sim_buf(a.device, a.dtype, a.shape)

    # thr = max(|a|,|b|)^2
    torch.abs(a, out=thr)
    torch.abs(b, out=sim)
    torch.maximum(thr, sim, out=thr)
    thr.mul_(thr)

    # sim = ((a*b)/thr + 1) * beta/2
    torch.mul(a, b, out=sim)
    sim.div_(thr)
    sim.add_(1.0).mul_(b2)
    torch.nan_to_num_(sim, nan=float(beta))

    # out = a + alpha*(b-c)
    torch.sub(b, c, out=out)
    out.mul_(float(alpha)).add_(a)

    # thr = a*(1-a/2) + b*(a/2)
    torch.mul(a, (1.0 - a2), out=thr)
    thr.add_(b, alpha=a2)

    # out = lerp(out, thr, sim)
    torch.lerp(out, thr, sim, out=out)

    out = _match_mean_std_like_a(out, a)
    return out.to(dtype=a_orig_dtype)

def _rand_like_compat(ref: torch.Tensor, *, dtype=torch.float32, generator=None) -> torch.Tensor:
    if generator is None:
        return torch.rand(ref.shape, device=ref.device, dtype=dtype)
    return torch.rand(ref.shape, device=ref.device, dtype=dtype, generator=generator)

def dare_merge(theta0, theta1, alpha, beta, generator=None):
    # match the shapes by padding with zeros
    if theta0.dim() in (1, 2):
        if theta0.dim() == 1:
            d = theta1.shape[0] - theta0.shape[0]
            if d > 0:
                theta0 = F.pad(theta0, (0, d))
            elif d < 0:
                theta1 = F.pad(theta1, (0, -d))
        else:  # dim == 2
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

    a = float(alpha)
    b = float(beta)
    denom = max(1.0 - b, 1e-6)

    delta = theta1 - theta0

    mask = _rand_like_compat(delta, dtype=torch.float32, generator=generator) < b
    scaled = (delta / denom)
    scaled = scaled * mask.to(delta.dtype)
    
    return torch.add(theta0, scaled.to(theta0.dtype), alpha=a)

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
    # alpha: mix strength
    # beta : fraction of elements to take (Top-k)
    diff = (b - a)
    d = diff.detach().float().abs().view(-1)
    n = d.numel()
    if n == 0:
        return a

    # k = number of elements to replace
    k = int(n * float(beta))
    if k <= 0:
        return a
    if k >= n:
        # replace all (alpha controls full replace)
        return (a + float(alpha) * diff).to(a.dtype)

    # kthvalue is 1-indexed: threshold for top-k largest == (n-k+1)-th smallest
    kth = n - k + 1
    thresh = d.kthvalue(kth).values

    mask = d.view_as(diff).ge_(thresh).to(diff.dtype)
    out = a + float(alpha) * diff * mask
    return out.to(a.dtype)

def norm_dir_blend(a, b, alpha):
    a32 = a.detach().float().view(-1); b32 = b.detach().float().view(-1)
    an = a32.norm() + 1e-12; bn = b32.norm() + 1e-12
    au = a32 / an; bu = b32 / bn
    du = F.normalize((1 - alpha) * au + alpha * bu, dim=0)
    mag = (1 - alpha) * an + alpha * bn
    out = (du * mag).view_as(a)
    return out.to(a.dtype)

def channel_cosine_gate(a, b, alpha, beta, eps=1e-12):
    if a.dim() == 4:
        axis = (1, 2, 3)   # per-out-channel
    elif a.dim() == 2:
        axis = (1,)        # per-out-feature
    else:
        return (1 - float(alpha)) * a + float(alpha) * b

    a32 = a.detach().float()
    b32 = b.detach().float()

    num = (a32 * b32).sum(dim=axis)
    den = (
        torch.linalg.vector_norm(a32, ord=2, dim=axis) *
        torch.linalg.vector_norm(b32, ord=2, dim=axis) + eps
    )

    cos = (num / den).clamp_(-1.0, 1.0)

    g = ((1.0 - cos) * float(beta)).clamp_(0.0, 1.0)
    while g.dim() < a.dim():
        g = g.unsqueeze(-1)

    mix = (1.0 - float(alpha)) * a + float(alpha) * b
    return (a * (1.0 - g) + mix * g).to(a.dtype)

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

    low  = (dist <= cut).to(A.dtype)
    high = 1 - low

    F = low * A + high * ((1 - float(alpha)) * A + float(alpha) * B)

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
    "COMP": (None,           None,                       "Save Components (model0 only)"),
    "CLIPXOR": (None,        None,                       "CLIP XOR (union-minus-intersection)"),
    "FWM":  (None,           feature_weighted_merge,     "Feature Weighted Merge"),
    "TF":  (None,            None,                       "Trim and Fill"),
}
modes_need_m2   = {"sAD", "AD", "TRS", "ST",  "TD", "SIM", "MD", "HUB"}
modes_need_beta = {"TRS", "ST", "TS",  "SIM", "MD", "DARE", "CHAN", "FREQ", "SPRSE"}

parser = argparse.ArgumentParser(description="Merge two or three models")

parser.add_argument("mode",         choices=list(theta_funcs.keys()),   help="Merging mode")
parser.add_argument("model_path",   type=str,                           help="Path to models")
parser.add_argument("model_0",      type=str,                           help="Name of model 0")
parser.add_argument("model_1",      type=str,                nargs="?", help="Optional, Name of model 1", default=None)
parser.add_argument("model_2",      type=str,                nargs="?", help="Optional, Name of model 2", default=None)

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
    "turbo":            "Apply delta (model_1 turbo, model_2 base) to model_0",
    "deturbo":          "Remove turbo delta (model_1 turbo, model_2 base) from model_0",
}.items():
    parser.add_argument(f"--{flag}", action="store_true", help=helpmsg, required=False)

parser.add_argument("--seed",   type=int,   help="Random seed for stochastic modes (e.g., DARE)", default=None)
parser.add_argument("--rebasin",   type=int,   help="ReBasin iterations", default=None)
parser.add_argument("--vae",    type=str,   help="Path of VAE", default=None, required=False)
parser.add_argument("--memo",   type=str,   help="Additional info bake in metadata", default=None)
parser.add_argument("--fine",   type=str,   help="Finetune the given keys on model 0", default=None, required=False)
parser.add_argument("--output",             help="Output file name without extension", default="merged", required=False)
parser.add_argument("--device", type=str,   help="Device to use, defaults to cpu", default="cpu", required=False)
parser.add_argument("--cfg_sens", type=float, default=1.0,
    help="(SDXL) Post-scale UNet cross-attention (attn2) projections to make CFG more sensitive. 1.0=off. सुझ: 1.05-1.15")

parser.add_argument("--cfg_sens_targets", type=str, default="kv,out",
    help="Which attn2 projections to scale: q,k,v,out,kv,qkv,all. Default: kv,out")

parser.add_argument("--sat_boost", type=float, default=1.0,
    help="(SDXL) Multiply merge strength for saturation-related layers. 1.0=off. Use 2.0 to double.")

parser.add_argument("--sat_boost_side", choices=["alpha", "beta", "both"], default="alpha",
    help="Apply sat_boost to alpha/beta/both. Default: alpha")

parser.add_argument("--sat_boost_tags", type=str, default=None,
    help="Comma-separated XL block tags to treat as saturation-related (e.g. IN00,IN01,IN02,IN03,M00). If omitted, heuristic is used.")

parser.add_argument("--sat_profile", choices=["legacy", "safe_attn2_out"], default="legacy",
    help="How sat_boost is applied. legacy=old behavior. safe_attn2_out=only OUT-block attn2.to_v/to_out + capped delta.")

parser.add_argument("--sat_delta_cap_pct", type=float, default=0.0,
    help="Cap the per-tensor delta magnitude by percentile (e.g. 99.5). 0=off. Helps prevent geometry break.")

parser.add_argument("--sat_boost_mix", type=float, default=1.0,
    help="Blend between normal and boosted result on sat targets. 1.0=fully boosted, 0.0=no effect. Suggest 0.3-0.8.")

parser.add_argument("--boost_clamp", choices=["auto", "clamp01", "none"], default="auto",
    help="Clamp boosted strengths for unstable modes. auto clamps for WS/TRS/ST/TS/DARE/CHAN/FREQ/SPRSE/MD/SIM etc.")

parser.add_argument("--vae_sat", type=float, default=1.0,
    help="Apply RGB saturation scaling inside VAE output (decoder.conv_out). 1.0=off. >1 more saturation, <1 less.")

args = parser.parse_args()
if args.mode not in {"NoIn", "RM", "SWAP", "CLIPXOR", "COMP"} and args.model_1 is None:
    raise SystemExit(f"mode '{args.mode}' needs model_1")

if args.save_quarter and args.save_half:
    print("[warn] --save_half and --save_quarter are both set; prioritizing --save_quarter (fp8).")
    args.save_half = False

if args.turbo and args.deturbo:
    raise SystemExit("--turbo and --deturbo cannot be used together")
turbo_convert = bool(args.turbo or args.deturbo)
if turbo_convert and (args.model_1 is None or args.model_2 is None):
    raise SystemExit("--turbo/--deturbo require model_1 and --model_2 (B=turbo, C=base)")

device = args.device
mode = args.mode
if mode in modes_need_m2 and (args.model_2 is None):
    raise SystemExit(f"mode '{mode}' needs 3rd model")
theta_func1, theta_func2, merge_name = theta_funcs[mode]
bake_vae_enabled = (args.vae is not None)

if mode not in ["SWAP", "CLIPXOR", "COMP"] and not turbo_convert:
    args.alpha, deep_a, block_a = wgt(args.alpha, [])
    args.beta,  deep_b, block_b = wgt(args.beta, [])
    useblocks = block_a or block_b
else:
    useblocks = False
    deep_a = deep_b = []

cosine0 = bool(args.cosine0)
cosine1 = bool(args.cosine1)
cosine2 = bool(args.cosine2)

cos_flags = [cosine0, cosine1, cosine2]
if sum(1 for f in cos_flags if f) > 1:
    raise SystemExit("cosine0, cosine1 and cosine2 cannot be posed at same time, choose one only")

if cosine2 and (args.model_2 is None):
    raise SystemExit("--cosine2 cannot be used when there are only 2 models given")

if cosine0 or cosine1 or cosine2:
    if mode not in {"WS", "ST", "TRS"}:
        raise SystemExit("--cosine0/--cosine1/--cosine2 are supported only for modes WS, ST, TRS")
    if cosine2 and mode == "WS":
        raise SystemExit("--cosine2 is only supported for ST/TRS (not WS)")
    if mode == "WS" and not (cosine0 ^ cosine1):
        raise SystemExit("WS with cosine requires exactly one of --cosine0 or --cosine1")
    
output_name = args.output
output_file = f"{output_name}.{'safetensors' if args.save_safetensors else 'ckpt'}"
output_path = normalize_path(os.path.join(args.model_path, output_file))
set_cache_filename(os.path.join(base_path(), "cache.json"))
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
comp_components = None

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

arch, theta_0 = detect_arch(theta_0)
theta_0 = upcast_fp8_state_dict(theta_0)

theta_1 = theta_2 = None
model_1_sha256 = model_2_sha256 = None

if mode not in ["NoIn", "COMP"]:
    interp_method = 0
    model_1_path = normalize_path(os.path.join(args.model_path, args.model_1))
    model_1_name = args.m1_name or stem(model_1_path)
    print(f"Loading {model_1_name}...")
    theta_1, model_1_sha256, model_1_hash, model_1_meta, cache_data = load_model(model_1_path, device, cache_data=cache_data)
    qd1 = qdtyper(theta_1)
    theta_1 = upcast_fp8_state_dict(theta_1)
    if mode == "SWAP":
        theta_1 = normalize_external_text_encoder(theta_1, arch)
    else:
        _, theta_1 = detect_arch(theta_1)
    if args.fine and not arch.get("ZI", False):
        fine = fineman([float(t) for t in args.fine.split(",")], arch)
    else:
        fine = ""
        
    if mode == "SWAP":
        components, only = _parse_components_with_only(str(args.alpha))
        
        if not components:
            components = {"unet", "vae", "clip-l", "clip-g", "clip", "transformer", "text", "text2"}

        moved, created, skipped, theta_0 = _swap_components_inplace(
            theta_0, theta_1,
            components,
            arch,
            src_only=only,
        )
        print(f"[SWAP] components={sorted(list(components))} only={sorted(list(only))}  moved:{moved}  created:{created}  shape_skipped:{skipped}")

        mode = "NoIn"
        theta_1 = None
        usebeta = False
        weights_a = weights_b = None
        alpha = beta = None
        
    elif mode in ["CLIPXOR", "XDARE"]:
        # --- in-place CLIPXOR / XDARE (no theta_res copy) ---

        base_hardness = 0.70
        hard_l = base_hardness
        hard_g = base_hardness
        hard_t5   = 0.60
        hard_clip = base_hardness

        # local arch flags for both models (avoid clobbering outer isxl/isflux/iszi)
        arch_a = detect_arch(theta_0)[0]
        arch_b = detect_arch(theta_1)[0]

        targets = _collect_clipxor_targets(theta_0, theta_1, arch=arch_a)

        suffix_pairs = []
        if not targets:
            suffix_pairs = _collect_clip_pairs_by_suffix(
                theta_0, theta_1,
                arch_a,
                arch_b
            )
            targets = [ka for (_, ka, _) in suffix_pairs]

        if not targets:
            print("[CLIPXOR] No eligible CLIP keys to merge (even after suffix matching). "
                "\nArchitectures may be incompatible or shapes differ.")
        else:
            suffix_to_kb = {ka: kb for (suf, ka, kb) in suffix_pairs} if suffix_pairs else {}

            # cache tier resolver for speed
            if arch_a.get("XL", False) or arch_b.get("XL", False):
                tier_fn = _clip_tier_for_xl
            elif arch_a.get("FLUX", False) or arch_b.get("FLUX", False):
                tier_fn = _clip_tier_for_flux
            elif arch_a.get("ZI", False) or arch_b.get("ZI", False):
                tier_fn = _clip_tier_for_zi
            else:
                tier_fn = None

            do_fine = bool('fine' in locals() and fine)
            semi_blend = _clipxor_semi_hard_blend

            for key_a in tqdm(targets, desc="CLIPXOR merging...", total=len(targets)):
                # resolve pair key in theta_1
                key_b = key_a if key_a in theta_1 else suffix_to_kb.get(key_a, None)
                if key_b is None:
                    continue

                A = theta_0.get(key_a, None)
                B = theta_1.get(key_b, None)
                if (A is None) or (B is None):
                    continue

                # hardness by tier
                hardness = base_hardness
                if tier_fn is not None:
                    tier = tier_fn(key_a)
                    if tier == "clip-l":
                        hardness = hard_l
                    elif tier == "clip-g":
                        hardness = hard_g
                    elif tier == "t5":
                        hardness = hard_t5
                    elif tier == "clip":
                        hardness = hard_clip
                    elif tier == "qwen3_4b":
                        hardness = hard_t5
                    elif tier == "cap_embedder":
                        hardness = hard_clip

                M_semi = semi_blend(
                    A, B,
                    hardness=float(hardness),
                    use_cosine_gate=True,
                    keep_stats=True
                )
                if do_fine:
                    M_semi = _finetune_inplace(key_a, M_semi, fine, arch=arch)

                # in-place writeback (no extra dict)
                theta_0[key_a] = M_semi

        # mode transition behavior
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
            weights_b, beta,  beta_info  = parse_ratio(args.beta,  beta_info,  deep_b)
    else:
        weights_a, alpha, alpha_info = parse_ratio(args.alpha, alpha_info, deep_a)
        if mode in modes_need_m2:
            model_2_path = normalize_path(os.path.join(args.model_path, args.model_2))
            model_2_name = args.m2_name or stem(model_2_path)
            print(f"Loading {model_2_name}...")
            theta_2, model_2_sha256, model_2_hash, model_2_meta, cache_data = load_model(model_2_path, device, cache_data=cache_data)
            qd2 = qdtyper(theta_2)
            _, theta_2 = detect_arch(theta_2)
            theta_2 = upcast_fp8_state_dict(theta_2)

        usebeta = mode in modes_need_beta
        if usebeta:
            weights_b, beta, beta_info = parse_ratio(args.beta, beta_info, deep_b)
        else:
            weights_b, beta = None, None
        if args.rebasin is not None:
            if arch.get("FLUX") or arch.get("ZI") or arch.get("AM"):
                print("[ReBasin] Unavailable architecture detected, skipping ReBasin (not supported).")
            else:
                print(f"[ReBasin] Running weight matching (Hungarian)... iter={args.rebasin}")
                ps = unet_permutation_spec(arch.get("XL", False))
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
    if args.mode == "COMP":
        atext = str(args.alpha).strip()
        if atext in {"", "0", "0.0", "none", "None"}:
            atext = "all"
        comp_components = _normalize_components_list(atext)

        if not comp_components:
            comp_components = {"unet", "vae", "clip-l", "clip-g", "clip", "transformer", "text", "text2"}

        before = len(theta_0)
        theta_0, kept, total = _filter_state_dict_by_components(theta_0, comp_components, arch)
        print(f"[COMP] components={sorted(list(comp_components))}  kept:{kept} / {before}")

        mode = "NoIn"
        theta_1 = None
        deep_a = deep_b = []
    usebeta = False 
    weights_a = weights_b = None
    alpha = beta = None
    arch = {t: False for t in arch.keys()}

if args.vae:
    if args.mode == "COMP" and comp_components is not None and ("vae" not in comp_components):
        print("[COMP] --vae was provided but 'vae' is not selected; skipping VAE bake.")
    else:
        vae_name = stem(args.vae)
        vae, *_ = load_model(normalize_path(args.vae), device, verify_hash=False)

if mode == "DARE":
    g = torch.Generator(device=device)
    if args.seed is not None:
        g.manual_seed(args.seed)

    def theta_func2_dare(a, b, cur_a, cur_b):
        return dare_merge(a, b, cur_a, cur_b, generator=g)

    theta_func2 = theta_func2_dare


# -----------------------------------------------------------------------------
# Cosine blend helpers
# -----------------------------------------------------------------------------

def _cosine_keys_intersection(base: dict, other: dict, vae_key: str, bake_vae_enabled: bool):
    skip = set(checkpoint_dict_skip_on_merge)
    keys = []
    for k in base.keys():
        if k in skip:
            continue
        if (not bake_vae_enabled) and (vae_key in k):
            continue
        if ("model" not in k and "text_encoders" not in k):
            continue
        if k not in other:
            continue

        a = base[k]
        b = other[k]
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            continue
        if (not a.is_floating_point()) or (not b.is_floating_point()):
            continue
        if a.shape != b.shape:
            continue
        keys.append(k)
    return keys


@torch.inference_mode()
def _cosine_combined_similarity_tensor(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a32 = a.detach().to(torch.float32)
    b32 = b.detach().to(torch.float32)

    # conv weight: [out, in, kh, kw] -> gate per out channel
    if a32.dim() == 4:
        a2 = a32.view(a32.shape[0], -1)
        b2 = b32.view(b32.shape[0], -1)
        cos = F.cosine_similarity(a2, b2, dim=1, eps=eps)  # [out]
        g = ((cos + 1.0) * 0.5).clamp_(0.0, 1.0).view(-1, 1, 1, 1)
        return g

    # linear: [out, in] -> gate per out channel
    if a32.dim() == 2:
        a2 = a32.view(a32.shape[0], -1)
        b2 = b32.view(b32.shape[0], -1)
        cos = F.cosine_similarity(a2, b2, dim=1, eps=eps)  # [out]
        g = ((cos + 1.0) * 0.5).clamp_(0.0, 1.0).view(-1, 1)
        return g

    # 1D or other: scalar gate
    av = a32.reshape(1, -1)
    bv = b32.reshape(1, -1)
    cos = F.cosine_similarity(av, bv, dim=1, eps=eps).squeeze(0)  # scalar
    g = ((cos + 1.0) * 0.5).clamp_(0.0, 1.0)
    while g.dim() < a.dim():
        g = g.unsqueeze(-1)
    return g

def _cosine_trimmed_minmax(samples_np: np.ndarray):
    samples_np = samples_np[np.isfinite(samples_np)]
    if samples_np.size == 0:
        return 0.0, 1.0

    p1  = float(np.percentile(samples_np, 1,  method="midpoint"))
    p99 = float(np.percentile(samples_np, 99, method="midpoint"))
    trimmed = samples_np[(samples_np >= p1) & (samples_np <= p99)]
    if trimmed.size == 0:
        trimmed = samples_np

    mn = float(np.min(trimmed))
    mx = float(np.max(trimmed))
    if abs(mx - mn) < 1e-6:
        mx = mn + 1e-6
    return mn, mx

@torch.inference_mode()
def _cosine_pair_stats(base: dict, other: dict, keys: list, sample_per_key: int = 32):
    samples = []
    for k in tqdm(keys, desc="Cosine Stage 0/2 (stats)"):
        cs = _cosine_combined_similarity_tensor(base[k], other[k])
        flat = cs.reshape(-1)
        if flat.numel() == 0:
            continue
        n = min(sample_per_key, flat.numel())
        idx = torch.linspace(0, flat.numel() - 1, steps=n, device=flat.device).to(torch.long)
        samp = flat.index_select(0, idx).detach().to("cpu").numpy()
        samples.append(samp)

    if not samples:
        return 0.0, 1.0

    all_s = np.concatenate(samples, axis=0).astype(np.float64, copy=False)
    return _cosine_trimmed_minmax(all_s)

@torch.inference_mode()
def _cosine_blend_tensor(a: torch.Tensor, b: torch.Tensor, strength: float, keep_stats: bool = True):
    # g: 似てるほど1、似てないほど0
    g = _cosine_combined_similarity_tensor(a, b)

    # alphaは「混合量」として素直に効かせる（thresholdにしない）
    mix = (abs(float(strength)) * g).clamp_(0.0, 1.0)

    out32 = torch.lerp(a.to(torch.float32), b.to(torch.float32), mix)

    # 色/トーンの事故を減らす（特にproj/ff等で有効）
    if keep_stats:
        out32 = _match_mean_std_like_a(out32, a)

    return out32.to(a.dtype)

def _strength_getter(alpha, weights, deep, blockids_local):
    def get(k: str) -> float:
        wi = _resolve_weight_index(k)

        # base strength
        cur = float(alpha) if alpha is not None else 0.0
        if (weights is not None) and wi > 0:
            cur = float(weights[wi - 1])

        # deep override (supports CLIP/LABEL/TIME/OUT)
        if deep:
            cur = elementals2(
                k, wi, deep, float(cur),
                blockids=blockids_local, arch=arch
            )
        return float(cur)
    return get

@torch.inference_mode()
def _cosine_merge_pair_inplace(
    base: dict,
    other: dict,
    strength_getter,
    vae_key: str,
    bake_vae_enabled: bool,
    sample_per_key: int = 32,
    fine=None,
    arch=None,
):
    keys = _cosine_keys_intersection(base, other, vae_key=vae_key, bake_vae_enabled=bake_vae_enabled)
    if not keys:
        return base

    for k in tqdm(keys, desc="Cosine Stage 1/2 (apply)"):
        cur = float(strength_getter(k))

        if _is_small_or_norm_or_bias(k, base[k]):
            cur_eff = cur * 0.25
            out = weighted_sum(base[k], other[k], cur_eff)
        else:
            out = _cosine_blend_tensor(base[k], other[k], cur, keep_stats=True)

        if fine:
            out = _finetune_inplace(k, out, fine, arch=arch)
        base[k] = out

    skip = set(checkpoint_dict_skip_on_merge)
    for k, v in other.items():
        if k in base:
            continue
        if k in skip:
            continue
        if (not bake_vae_enabled) and (vae_key in k):
            continue
        if ("model" not in k and "text_encoders" not in k):
            continue
        base[k] = v

    return base

def _pick_base_and_others_for_cosine(theta_0, theta_1, theta_2, cosine_sel: int):
    if cosine_sel == 0:
        base_idx, base_sd = 0, theta_0
        others = [(theta_1, "alpha"), (theta_2, "beta")]
    elif cosine_sel == 1:
        base_idx, base_sd = 1, theta_1
        others = [(theta_0, "alpha"), (theta_2, "beta")]
    else:
        base_idx, base_sd = 2, theta_2
        others = [(theta_0, "alpha"), (theta_1, "beta")]
    return base_idx, base_sd, others


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
    if arch.get("FLUX", False):
        theta_1, theta_2 = maybe_to_qdtype(theta_1, theta_2, qd1, qd2, device)
    diff_inplace(theta_1, theta_2, theta_func1, "Getting Difference of Model 1 and 2")
    del theta_2

if arch.get("FLUX", False):
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

ZI_WLEN = len(BLOCKIDZI) - 1  # 33
AM_WLEN = len(BLOCKIDAM) - 1  # 29

def _fit_weights_to_len(w, target_len: int):
    if w is None:
        return None
    w = list(w)
    if not w:
        return [0.0] * target_len

    if len(w) != target_len:
        x0 = np.arange(len(w))
        x1 = np.linspace(0, len(w) - 1, target_len)
        w = np.interp(x1, x0, np.asarray(w, dtype=np.float64)).tolist()

    if len(w) > target_len:
        w = w[:target_len]
    elif len(w) < target_len:
        w += [w[-1]] * (target_len - len(w))
    return w

def _fit_weights_for_am(w):
    return _fit_weights_to_len(w, AM_WLEN)

def _fit_weights_for_zi(w):
    return _fit_weights_to_len(w, ZI_WLEN)

if mode not in ["NoIn", "TF"]:
    if arch.get("XL", False) and useblocks:
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
    elif arch.get("ZI", False) and useblocks:
        print("Detected Zimage architecture.")
        weights_a = _fit_weights_for_zi(weights_a)
        weights_b = _fit_weights_for_zi(weights_b) if weights_b is not None else None
        # print(f"alpha weights for ZI: {weights_a}")
        # print(f"beta weights for ZI: {weights_b}")
    elif arch.get("AM", False) and useblocks:
        print("Detected Anima (AM) architecture.")
        weights_a = _fit_weights_for_am(weights_a)
        weights_b = _fit_weights_for_am(weights_b) if weights_b is not None else None
        
def _resolve_weight_index(key: str) -> int:
    block, tag = blockfromkey(key, arch=arch)
    if block == "Not Merge":
        return -1
    return _TAG2IDX.get(tag, -1)

def make_param_resolver(alpha, beta, weights_a, weights_b, deep_a, deep_b, blockids, usebeta: bool):
    def get(key: str):
        wi = _resolve_weight_index(key)

        # allow pseudo-tag keys even when _blockfromkey_cached returns Not Merge
        if wi < 0:
            tag = extra_tag_for_key(key, arch=arch)
            if tag is None:
                return None

        cur_a = alpha
        if weights_a is not None and wi > 0:
            cur_a = weights_a[wi - 1]
        if deep_a:
            cur_a = elementals2(
                key, wi, deep_a, float(cur_a),
                blockids=blockids, arch=arch
            )

        cur_b = None
        if usebeta:
            cur_b = beta
            if weights_b is not None and wi > 0:
                cur_b = weights_b[wi - 1]
            if deep_b:
                cur_b = elementals2(
                    key, wi, deep_b, float(cur_b),
                    blockids=blockids, arch=arch
                )

        return wi, cur_a, cur_b
    return get

def _parse_csv_set(s: str):
    return {x.strip() for x in s.split(",") if x.strip()}

def _tag_for_key_safe(key: str):
    # returns block tag like IN00, M00, OUTxx... if available
    block, tag = blockfromkey(key, arch=arch)
    if block == "Not Merge" or tag is None:
        tag = extra_tag_for_key(key, arch=arch)
    return tag

def _is_vae_key(key: str, vae_key: str):
    return (vae_key in key) or key.startswith("vae.") or key.startswith("first_stage_model.") or key.startswith("model.vae.") or key.startswith("model.first_stage_model.")

def _is_cfg_attn2_key(key: str):
    k = key.lower()
    # SDXL UNet cross-attn usually includes "attn2" in transformer blocks
    if "attn2" not in k:
        return False
    # avoid norms etc if you want stricter: keep projections only
    return any(p in k for p in (".to_q.", ".to_k.", ".to_v.", ".to_out.", ".proj_in.", ".proj_out.", "to_out.0."))

def _cfg_targets_match(key: str, targets_set: set[str]):
    k = key.lower()
    # normalize: if user says kv -> k,v
    if "all" in targets_set:
        return _is_cfg_attn2_key(key)
    want_q = ("q" in targets_set) or ("qkv" in targets_set)
    want_k = ("k" in targets_set) or ("kv" in targets_set) or ("qkv" in targets_set)
    want_v = ("v" in targets_set) or ("kv" in targets_set) or ("qkv" in targets_set)
    want_o = ("out" in targets_set)

    if "attn2" not in k:
        return False
    if want_q and ".to_q." in k: return True
    if want_k and ".to_k." in k: return True
    if want_v and ".to_v." in k: return True
    if want_o and (".to_out." in k or "to_out.0." in k): return True
    # proj_in/out are sometimes used by implementations; treat as out-ish
    if want_o and (".proj_out." in k): return True
    if want_q and (".proj_in." in k): return True
    return False

def _is_saturation_key(key: str, vae_key: str, sat_tags: set[str] | None):
    # Heuristic:
    # 1) VAE decoder-ish keys
    if _is_vae_key(key, vae_key):
        kl = key.lower()
        # focus on decoder & quant conv (color response is often here)
        if ("decoder" in kl) or ("post_quant" in kl) or ("quant_conv" in kl):
            return True
        # if user really wants broad VAE boosting, allow through tags (sat_tags=None => heuristic only)
        return False

    # 2) UNet early blocks by tag (if available)
    tag = _tag_for_key_safe(key)
    if sat_tags is not None and tag in sat_tags:
        return True

    # 3) Fallback key-pattern for early convs
    kl = key.lower()
    if any(p in kl for p in ("conv_in", "conv_out", "input_blocks.0", "input_blocks.1", "input_blocks.2")):
        return True

    return False

def _clamp_for_mode(mode: str, a: float, b: float | None):
    if args.boost_clamp == "none":
        return a, b

    clamp01 = (args.boost_clamp == "clamp01")
    auto = (args.boost_clamp == "auto")

    def c01(x): 
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))

    # modes where alpha/beta should be in [0,1] to avoid nonsense
    if clamp01 or (auto and mode in {"WS","SIG","GEO","MAX","ST","TRS","TS","DARE","CHAN","FREQ","SPRSE","SIM","MD"}):
        a = c01(a)
        if b is not None:
            b = c01(b)

        # TRS expects a+b<=1 typically
        if mode == "TRS" and b is not None:
            s = a + b
            if s > 1.0 and s > 1e-12:
                a = a / s
                b = b / s

    return a, b

def apply_merge_strength_boosts(key: str, cur_a: float, cur_b: float | None, mode: str, vae_key: str):
    # sat tags default for SDXL if user didn't specify
    sat_tags = None
    if args.sat_boost_tags:
        sat_tags = _parse_csv_set(args.sat_boost_tags)
    else:
        # safe-ish default: early-ish tags (you can refine later)
        sat_tags = {"IN00","IN01","IN02","IN03","IN04","IN05","M00"}
        
    if _is_vae_key(key, vae_key):
        return _clamp_for_mode(mode, float(cur_a), (float(cur_b) if cur_b is not None else None))

    # ---- SAFE PROFILE: only OUT attn2 v/out ----
    if args.sat_profile == "safe_attn2_out":
        if args.sat_boost != 1.0 and _is_outblock_attn2_vout_key(key):
            if args.sat_boost_side in {"alpha","both"}:
                cur_a *= float(args.sat_boost)
            if cur_b is not None and args.sat_boost_side in {"beta","both"}:
                cur_b *= float(args.sat_boost)

        cur_a, cur_b = _clamp_for_mode(mode, float(cur_a), (float(cur_b) if cur_b is not None else None))
        return cur_a, cur_b

    # saturation boost
    if args.sat_boost != 1.0 and _is_saturation_key(key, vae_key=vae_key, sat_tags=sat_tags):
        if args.sat_boost_side in {"alpha","both"}:
            # avoid over-boosting tiny/norm/bias
            if _is_small_or_norm_or_bias(key, theta_0.get(key, torch.empty(0))):
                cur_a *= (1.0 + (args.sat_boost - 1.0) * 0.35)
            else:
                cur_a *= args.sat_boost
        if cur_b is not None and args.sat_boost_side in {"beta","both"}:
            cur_b *= args.sat_boost

    # (optional) you can also boost CFG-related layers pre-merge here if you want:
    # if args.cfg_boost != 1.0 and _is_cfg_attn2_key(key): ...

    cur_a, cur_b = _clamp_for_mode(mode, float(cur_a), (float(cur_b) if cur_b is not None else None))
    return cur_a, cur_b

@torch.inference_mode()
def apply_cfg_sens_inplace(sd: dict, gain: float, targets: str):
    if abs(float(gain) - 1.0) < 1e-12:
        return sd
    tset = _parse_csv_set(targets.lower())
    # allow shorthand like "kv,out"
    # already handled by _cfg_targets_match
    scaled = 0
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor) or (not v.is_floating_point()):
            continue
        if not _cfg_targets_match(k, tset):
            continue
        sd[k] = v.mul(float(gain))
        scaled += 1
    print(f"[cfg_sens] scaled {scaled} tensors (gain={gain}, targets={targets})")
    return sd

def _is_outblock_attn2_vout_key(key: str):
    tag = _tag_for_key_safe(key)  # already in your code
    if tag is None or (not str(tag).startswith("OUT")):
        return False
    kl = key.lower()
    if "attn2" not in kl:
        return False
    # only value/out projections
    if (".to_v." in kl) or (".to_out." in kl) or ("to_out.0." in kl):
        return True
    return False

@torch.inference_mode()
def _cap_delta_percentile(delta: torch.Tensor, pct: float):
    pct = float(pct)
    if pct <= 0.0 or pct >= 100.0:
        return delta
    d = delta.detach().float().abs().reshape(-1)
    if d.numel() == 0:
        return delta
    # kthvalue: k-th smallest (1-indexed)
    k = int(d.numel() * (pct / 100.0))
    k = max(1, min(k, d.numel()))
    thr = d.kthvalue(k).values
    return delta.clamp(min=-thr, max=thr)

cosine_applied = False

use_cos0 = bool(args.cosine0)
use_cos1 = bool(args.cosine1)
use_cos2 = bool(args.cosine2)
cosine_sel = None if (not any([use_cos0, use_cos1, use_cos2])) else (0 if use_cos0 else (1 if use_cos1 else 2))
blockids = (
    BLOCKIDFLUX if arch.get("FLUX", False) else
    (BLOCKIDXLL if arch.get("XL", False) else
     (BLOCKIDZI if arch.get("ZI", False) else
      (BLOCKIDAM if arch.get("AM", False) else BLOCKID)))
)
_TAG2IDX = {t: i for i, t in enumerate(blockids)}

if cosine_sel is not None:
    vae_key_local = "first_stage_model" if not (arch.get("FLUX", False) or arch.get("ZI", False)) else "vae"

    base_idx, base_sd, others = _pick_base_and_others_for_cosine(theta_0, theta_1, theta_2, cosine_sel)

    if mode == "WS":
        others = [(sd, tag) for (sd, tag) in others if (sd is not None and tag == "alpha")]
    else:
        others = [(sd, tag) for (sd, tag) in others if (sd is not None)]

    getA = _strength_getter(alpha, weights_a, deep_a, blockids)
    getB = _strength_getter(beta,  weights_b, deep_b, blockids) if (beta is not None) else None

    for other_sd, tag in others:
        sg = getA if tag == "alpha" else getB
        if sg is None:
            continue
        _cosine_merge_pair_inplace(
            base_sd, other_sd,
            strength_getter=sg,
            vae_key=vae_key_local,
            bake_vae_enabled=bake_vae_enabled,
            sample_per_key=32,
            fine=fine,
        )

    theta_0 = base_sd

    cosine_applied = True
    mode = "NoIn"
    theta_1 = None
    theta_2 = None
    usebeta = False
    weights_a = weights_b = None
    alpha = beta = None
    deep_a = deep_b = []

if turbo_convert:
    # ensure refs are loaded
    # B = model_1 (turbo), C = model_2 (base)
    # if deturbo, sign = -1 else +1

    # (load theta_1, theta_2 as usual; make sure you DO load both)
    # (optional) arch check: detect_arch(theta_1/theta_2) matches A

    # parse alpha/weights like usual (default alpha=1.0 recommended for full convert)
    if str(args.alpha).strip() in {"", "0", "0.0"}:
        args.alpha = 1.0

    args.alpha, deep_a, block_a = wgt(args.alpha, [])
    weights_a, alpha, alpha_info = parse_ratio(args.alpha, "", deep_a)

    blockids = (
        BLOCKIDFLUX if arch.get("FLUX", False) else
        (BLOCKIDXLL if arch.get("XL", False) else
        (BLOCKIDZI if arch.get("ZI", False) else
        (BLOCKIDAM if arch.get("AM", False) else BLOCKID)))
    )
    _TAG2IDX = {t: i for i, t in enumerate(blockids)}

    vae_key = "first_stage_model" if not (arch.get("FLUX", False) or arch.get("ZI", False)) else "vae"
    resolver = make_param_resolver(alpha, None, weights_a, None, deep_a, [], blockids, usebeta=False)

    do_fine = bool('fine' in locals() and fine)
    theta_0 = turbo_convert_inplace(
        theta_0, theta_1, theta_2, resolver,
        deturbo=bool(args.deturbo),
        vae_key=vae_key,
        bake_vae_enabled=bake_vae_enabled,
        fine=(fine if do_fine else None),
    )

    # stop normal merge path
    mode = "NoIn"
    theta_1 = theta_2 = None
    usebeta = False
    weights_a = weights_b = None
    alpha = beta = None
    deep_a = deep_b = []

@torch.inference_mode()
def build_merge_keys(theta_0, theta_1, theta_2, merge_cache, mode, usebeta, vae_key, bake_vae_enabled):
    if theta_1 is None:
        return []

    t1 = theta_1
    t2 = theta_2
    skip = checkpoint_dict_skip_on_merge

    keys = []
    for k in merge_cache.keys():
        if (not bake_vae_enabled) and (vae_key in k):
            continue
        if k in skip:
            continue
        if ("model" not in k and "text_encoders" not in k):
            continue
        if k not in t1:
            continue
        if (mode != "DARE") and (usebeta or mode == "TD") and (t2 is not None) and (k not in t2):
            continue
        keys.append(k)
    return keys


def remerge_model(target_dict, source_dict, desc, mode, resolver, theta_2=None):
    t = target_dict
    s = source_dict
    skip = checkpoint_dict_skip_on_merge

    for key in tqdm(s.keys(), desc=desc, total=len(s)):
        if arch.get("FLUX", False):
            continue
        if key in skip:
            continue
        if ("model" not in key and "text_encoders" not in key):
            continue
        if key in t:
            continue

        if mode in {"TRS", "ST"} and theta_2 is not None and key in theta_2:
            ent = resolver(key)
            if ent is None:
                t[key] = s[key]
                continue
            _, _, cur_b = ent
            try:
                t[key] = torch.lerp(s[key], theta_2[key], float(cur_b))
            except Exception:
                t[key] = s[key]
        else:
            t[key] = s[key]
    return t

if mode not in ["NoIn", "TF"]:
    with torch.inference_mode():
        vae_key = "first_stage_model" if not (arch.get("FLUX", False) or arch.get("ZI", False)) else "vae"
        resolver = make_param_resolver(alpha, beta, weights_a, weights_b, deep_a, deep_b, blockids, usebeta)
        
        cache_skip = checkpoint_dict_skip_on_merge
        func = theta_func2
        do_fine = bool('fine' in locals() and fine)
        key_list = list(theta_0.keys())
        for key in tqdm(key_list, desc=f"{merge_name} Merging...", total=len(theta_0)):
            if (not bake_vae_enabled) and (vae_key in key):
                continue
            if key in cache_skip:
                continue
            if ("model" not in key and "text_encoders" not in key):
                continue
            if theta_1 is None or (key not in theta_1):
                continue
            if (mode != "DARE") and (usebeta or mode == "TD") and (theta_2 is not None) and (key not in theta_2):
                continue

            ent = resolver(key)
            if ent is None:
                continue
            _, cur_a, cur_b = ent

            is_sat_target = (args.sat_profile == "safe_attn2_out" and _is_outblock_attn2_vout_key(key))

            if is_sat_target and (float(args.sat_boost_mix) < 1.0 or float(args.sat_delta_cap_pct) > 0.0):
                # 1) normal (no sat boost)
                cur_a0, cur_b0 = _clamp_for_mode(mode, float(cur_a), (float(cur_b) if cur_b is not None else None))

                # 2) boosted
                cur_a1, cur_b1 = apply_merge_strength_boosts(key, cur_a, cur_b, mode=mode, vae_key=vae_key)

                # compute both
                if usebeta and mode in modes_need_m2:
                    out0 = func(ad, b, theta_2[key], cur_a0, cur_b0)
                    out1 = func(ad, b, theta_2[key], cur_a1, cur_b1)
                elif usebeta:
                    out0 = func(ad, b, cur_a0, cur_b0)
                    out1 = func(ad, b, cur_a1, cur_b1)
                else:
                    out0 = func(ad, b, cur_a0)
                    out1 = func(ad, b, cur_a1)

                mix = float(args.sat_boost_mix)
                out = torch.lerp(out0.to(torch.float32), out1.to(torch.float32), mix).to(out0.dtype)

                if float(args.sat_delta_cap_pct) > 0.0:
                    delta = (out.to(torch.float32) - a.to(torch.float32))
                    delta = _cap_delta_percentile(delta, float(args.sat_delta_cap_pct))
                    out = (a.to(torch.float32) + delta).to(a.dtype)

            else:
                cur_a, cur_b = apply_merge_strength_boosts(
                    key, cur_a, cur_b,
                    mode=mode,
                    vae_key=vae_key
                )

            a = theta_0[key]
            b = theta_1[key]
            
            if (not isinstance(a, torch.Tensor)) or (not isinstance(b, torch.Tensor)):
                continue
            if (not a.is_floating_point()) or (not b.is_floating_point()):
                continue
            if a.shape != b.shape:
                continue
            if usebeta and (mode in modes_need_m2) and (usebeta or mode == "TD"):
                c = theta_2[key]
                if (not isinstance(c, torch.Tensor)) or (not c.is_floating_point()) or (c.shape != a.shape):
                    continue

            if mode == "sAD":
                bf = b.detach().float().cpu()
                filt = scipy.ndimage.gaussian_filter(bf.numpy(), sigma=1)
                out = a + cur_a * torch.from_numpy(filt).to(a.device, dtype=a.dtype)
                theta_0[key] = _finetune_inplace(key, out, fine, arch=arch) if do_fine else out
                continue

            if mode == "TD":
                t1f = b.float()
                t2f = theta_2[key].float()
                if torch.equal(t1f, t2f):
                    continue
                t0f = a.float()
                diff = (t1f - t2f)
                absdiff = diff.abs()
                distA0 = (t1f - t0f).abs()
                denom = distA0 + absdiff
                scale = torch.where(denom != 0, distA0 / denom, torch.zeros((), device=t0f.device))
                scale = diff.sign() * scale.abs()
                out = (t0f + (scale * absdiff) * (float(cur_a) * 1.8)).to(a.dtype)
                theta_0[key] = _finetune_inplace(key, out, fine, arch=arch) if do_fine else out
                continue

            if mode == "TS":
                if a.dim() == 0:
                    continue
                n = a.shape[0]
                if cur_a + cur_b <= 1:
                    s, e = int(n * cur_b), int(n * (cur_a + cur_b))
                    theta_0[key][s:e, ...].copy_(b[s:e, ...])
                else:
                    s, e = int(n * (cur_a + cur_b - 1)), int(n * cur_b)
                    t = b.clone()
                    t[s:e, ...].copy_(a[s:e, ...])
                    theta_0[key] = t
                theta_0[key] = _finetune_inplace(key, theta_0[key], fine, arch=arch) if do_fine else theta_0[key]
                continue

            if (a.shape != b.shape) and (a.dim() == 4) and (b.dim() == 4) and (a.shape[0] == b.shape[0]) and (a.shape[2:] == b.shape[2:]):
                use = min(a.shape[1], b.shape[1], 4)
                ad = a[:, :use, ...]
            else:
                ad = a

            if usebeta and mode in modes_need_m2:
                out = func(ad, b, theta_2[key], cur_a, cur_b)
            elif usebeta:
                out = func(ad, b, cur_a, cur_b)
            else:
                out = func(ad, b, cur_a)

            theta_0[key] = _finetune_inplace(key, out, fine, arch=arch) if do_fine else out

    if mode != "DARE":
        if mode != "AD":
            theta_0 = remerge_model(theta_0, theta_1, "Remerging...", mode, resolver, theta_2=theta_2)
        else:
            theta_0 = remerge_model(theta_0, theta_1, "Remerging...", mode, resolver)
    del theta_1
    try:
        if theta_2:
            theta_0 = remerge_model(theta_0, theta_2, desc="Remerging...", mode=mode, resolver=resolver)
            del theta_2
    except NameError:
        pass

else:
    if args.mode == "TF":
        theta_0 = prune_extras_vs_model1(theta_0, theta_1)
        resolver = make_param_resolver(alpha, beta, weights_a, weights_b, deep_a, deep_b, blockids, usebeta)
        theta_0 = remerge_model(theta_0, theta_1, desc="Remerging...", mode=mode, resolver=resolver, theta_2=theta_2)
    arch, theta_0 = detect_arch(theta_0)
    vae_key = "first_stage_model" if not (arch.get("FLUX", False) or arch.get("ZI", False)) else "vae"
    if (not cosine_applied) and args.fine and not arch.get("ZI", False):
        fine = fineman([float(t) for t in args.fine.split(",")], arch=arch)
        for key in tqdm(theta_0.keys(), desc="Fine Tuning ..."):
            if args.vae is None and vae_key in key:
                continue
            theta_0[key] = _finetune_inplace(key, theta_0[key], fine, arch=arch)
    elif not cosine_applied:
        fine = ""
        
def _strip_vae_root(k: str):
    for r in ("vae.", "first_stage_model.", "model.vae.", "model.first_stage_model."):
        if k.startswith(r):
            return k[len(r):]
    return k

if args.vae:
    for k in tqdm(vae.keys(), desc=f"Baking in VAE[{vae_name}] ..."):
        tk = vae_key + "." + _strip_vae_root(k)
        theta_0[tk] = to_half(vae[k], args.save_half)
    del vae
    
    if float(args.vae_sat) != 1.0:
        theta_0 = apply_vae_saturation_inplace(theta_0, vae_key=vae_key, sat=float(args.vae_sat))

arch, theta_0 = detect_arch(theta_0)

if arch.get("XL", False):
    for k in tqdm([k for k in theta_0.keys() if "cond_stage_model." in k], desc="Cond resolving..."):
        del theta_0[k]

if float(args.cfg_sens) != 1.0:
    if not arch.get("XL", False):
        print("[cfg_sens] Warning: --cfg_sens is tuned for SDXL; applying anyway.")
    theta_0 = apply_cfg_sens_inplace(theta_0, gain=float(args.cfg_sens), targets=str(args.cfg_sens_targets))

theta_0 = prepare_state_dict_for_save(
    theta_0,
    args=args,
    arch=arch,
    vae_prefix=vae_key,
    prune=bool(args.prune),
    make_cpu=True,
    make_contiguous=True,
)

metadata = {"format": "safetensors" if args.save_safetensors else "ckpt", "sd_merge_models": {}, "sd_merge_recipe": None}
if args.memo is not None:
    metadata["memo"] = args.memo

calcs = [
    name for flag, name in [
        (bool(args.cosine0),  "cosine_0"),
        (bool(args.cosine1),  "cosine_1"),
        (bool(args.cosine2),  "cosine_2"),
        (args.use_dif_10,    "use_dif_10"),
        (args.use_dif_20,    "use_dif_20"),
        (args.use_dif_21,    "use_dif_21"),
    ] if flag
]
if args.fine:
    calcs.append(f"fine[{fine}]")
calcl = ",".join(calcs) or None

fp = "fp8" if args.save_quarter else ("fp16" if args.save_half else ("bf16" if args.save_bhalf else "fp32"))

merge_recipe = {
    "type":                 "merge-models-chattiori",
    "primary_model_hash":   model_0_sha256,
    "secondary_model_hash": model_1_sha256 if (model_1_sha256 is not None) else None,
    "tertiary_model_hash":  model_2_sha256 if (model_2_sha256 is not None) else None,
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
elif args.mode == "COMP":
    merge_recipe["comp_components_alpha_text"] = str(args.alpha)
elif args.rebasin is not None:
    merge_recipe["rebasin"] = {
        "iter": args.rebasin,
        "min_channels": 64,
        "max_channels": 4096,
    }
    
if float(args.cfg_sens) != 1.0:
    calcs.append(f"cfg_sens[{args.cfg_sens}|{args.cfg_sens_targets}]")
if float(args.sat_boost) != 1.0:
    calcs.append(f"sat_boost[{args.sat_boost}|{args.sat_boost_side}|{args.sat_boost_tags or 'auto'}]")
    
metadata["sd_merge_recipe"] = json.dumps(merge_recipe)

def add_model_metadata(s256, hashed, meta, model_name):
    metadata["sd_merge_models"][s256] = {
        "name": model_name,
        "legacy_hash": hashed,
        "sd_merge_recipe": meta.get("sd_merge_recipe"),
    }
    metadata["sd_merge_models"].update(meta.get("sd_merge_models", {}))

add_model_metadata(model_0_sha256, model_0_hash, model_0_meta, model_0_name)
if model_1_sha256 is not None:
    add_model_metadata(model_1_sha256, model_1_hash, model_1_meta, model_1_name)
if model_2_sha256 is not None:
    add_model_metadata(model_2_sha256, model_2_hash, model_2_meta, model_2_name)

metadata["sd_merge_models"] = json.dumps(metadata["sd_merge_models"])

delete_targets = []
if args.delete_source:
    for p, cond in [
        (os.path.join(args.model_path, args.model_0), True),
        (os.path.join(args.model_path, args.model_1), model_1_sha256 is not None),
        (os.path.join(args.model_path, args.model_2), model_2_sha256 is not None),
    ]:
        if cond and os.path.isfile(p):
            delete_targets.append(p)

merge_success = False
try:
    print(f"Saving as {output_file}...")
    if args.save_safetensors:
        safetensors.torch.save_file(
            theta_0, output_path,
            metadata=None if args.no_metadata else metadata
        )
    else:
        torch.save({"state_dict": theta_0}, output_path, _use_new_zipfile_serialization=False)

    merge_success = True
    print(f"Done! ({round(os.path.getsize(output_path)/1073741824, 2)}G)")
except Exception as e:
    print("ERROR while saving:", repr(e))
finally:
    if args.delete_source and merge_success:
        for p in delete_targets:
            try:
                os.remove(p)
                print(f"[delete_source] Removed source: {p}")
            except Exception as e:
                print(f"[delete_source] Failed to remove {p}: {e}")