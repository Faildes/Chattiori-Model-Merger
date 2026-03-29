import torch
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

try:
    FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)
    FP8_E5M2 = getattr(torch, "float8_e5m2", None)
    FP8_DTYPES = tuple(d for d in (FP8_E4M3, FP8_E5M2) if d is not None)
except Exception:
    FP8_E4M3 = FP8_E5M2 = None
    FP8_DTYPES = ()
    
def _common_dtype(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    dt = torch.promote_types(torch.promote_types(a.dtype, b.dtype), c.dtype)
    if dt in FP8_DTYPES:
        dt = torch.float16
    return dt

def trim_delta(delta: torch.Tensor, percentile: float = 0.5) -> torch.Tensor:
    if delta.dim() == 4 and min(delta.shape[-2:]) > 2:
        blurred = F.avg_pool2d(delta, kernel_size=3, stride=1, padding=1)
        delta = delta * (1 - percentile) + blurred * percentile
    return delta

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
modes_need_beta = {"TRS", "ST", "TS",  "SIM", "MD", "DARE", "XDARE", "CHAN", "FREQ", "SPRSE"}
