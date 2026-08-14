from __future__ import annotations
import os
import re
import numpy as np
import json
import argparse
import torch
import torch.nn.functional as F
import scipy.ndimage
from tqdm.auto import tqdm
from collections import OrderedDict

from Utils import (
    wgt, rand_ratio, parse_ratio, maybe_to_qdtype, diff_inplace,
    fineman, weighttoxl, BLOCKID, BLOCKIDFLUX, BLOCKIDXLL, BLOCKIDZI, BLOCKIDAM28, BLOCKIDK2,
    blockfromkey, checkpoint_dict_skip_on_merge, elementals2, extra_tag_for_key, _is_small_or_norm_or_bias,
    to_half, base_path, merge_cache_json, detect_arch,
    _swap_components_inplace, _normalize_components_list, _finetune_inplace,
    _clip_tier_for_xl, _clip_tier_for_flux, _clip_tier_for_zi, _clipxor_semi_hard_blend,
    _collect_clipxor_targets, _collect_clip_pairs_by_suffix, turbo_convert_inplace,
    normalize_path, prune_extras_vs_model1, unet_permutation_spec,
    weight_matching, apply_permutation, _parse_components_with_only,
    _filter_state_dict_by_components, apply_vae_saturation_inplace,
    normalize_external_text_encoder, _model_path, _load_umodel, _model_stem, _build_output_path, 
    _clone_info, _register_merge_parents, _save_umodel,
    configure_save_precision, ANIMA_BASE_BLOCK_COUNT, ANIMA29_BLOCK_COUNT,
    detect_anima_block_count, anima_blockids_for_arch, remap_anima_theta, remap_anima_weights
)

from model import UnifiedModel

from merge_modes import theta_funcs, modes_need_m2, modes_need_beta, dare_merge, _match_mean_std_like_a, weighted_sum, get_difference


_LEGACY_MODE_ALIASES = {
    "CLIPXOR": ("NoIn", True),
    "XDARE": ("DARE", True),
}

_COMPONENT_SYNONYMS = {
    "u": "unet", "unet": "unet",
    "v": "vae", "vae": "vae",
    "clip": "clip", "te": "clip",
    "clip-l": "clip-l", "clipl": "clip-l", "clip_l": "clip-l", "l": "clip-l", "text-l": "clip-l",
    "clip-g": "clip-g", "clipg": "clip-g", "clip_g": "clip-g", "g": "clip-g", "text-g": "clip-g",
    "denoiser": "transformer", "denoise": "transformer", "transformer": "transformer", "mmdit": "transformer",
    "t5": "text", "t5-xxl": "text", "text": "text", "text1": "text", "text2": "text2",
    "all": "all",
}
_ALL_COMPONENTS = {"unet", "vae", "clip-l", "clip-g", "clip", "transformer", "text", "text2"}


def _parse_save_component_values(values) -> set[str]:
    """Parse repeatable --save-component values without silently accepting typos."""
    if not values:
        return set()
    tokens = []
    for value in values:
        tokens.extend(t.strip().lower() for t in str(value).replace(";", ",").split(",") if t.strip())
    unknown = sorted({t for t in tokens if t not in _COMPONENT_SYNONYMS})
    if unknown:
        raise ValueError(
            "Unknown --save-component value(s): " + ", ".join(unknown) +
            ". Supported values: unet, vae, clip, clip-l, clip-g, transformer, text, text2, all."
        )
    mapped = {_COMPONENT_SYNONYMS[t] for t in tokens}
    return set(_ALL_COMPONENTS) if "all" in mapped else mapped


@torch.inference_mode()
def _apply_clipxor_inplace(model0: UnifiedModel, model1: UnifiedModel, *, fine=None, arch=None) -> int:
    """Apply the former CLIPXOR mode as a pre-merge option."""
    base_hardness = 0.70
    hard_l = base_hardness
    hard_g = base_hardness
    hard_t5 = 0.60
    hard_clip = base_hardness

    arch_a = model0.arch
    arch_b = model1.arch
    targets = _collect_clipxor_targets(model0.theta, model1.theta, arch=arch_a)

    suffix_pairs = []
    if not targets:
        suffix_pairs = _collect_clip_pairs_by_suffix(model0.theta, model1.theta, arch_a, arch_b)
        targets = [ka for (_, ka, _) in suffix_pairs]

    if not targets:
        print(
            "[CLIPXOR] No eligible CLIP keys to merge (even after suffix matching).\n"
            "Architectures may be incompatible or shapes differ."
        )
        return 0

    suffix_to_kb = {ka: kb for (_, ka, kb) in suffix_pairs} if suffix_pairs else {}
    if arch_a.get("XL", False) or arch_b.get("XL", False):
        tier_fn = _clip_tier_for_xl
    elif arch_a.get("FLUX", False) or arch_b.get("FLUX", False):
        tier_fn = _clip_tier_for_flux
    elif arch_a.get("ZI", False) or arch_b.get("ZI", False):
        tier_fn = _clip_tier_for_zi
    else:
        tier_fn = None

    merged_count = 0
    for key_a in tqdm(targets, desc="CLIPXOR pre-merge...", total=len(targets)):
        key_b = key_a if key_a in model1.theta else suffix_to_kb.get(key_a)
        if key_b is None:
            continue
        a = model0.theta.get(key_a)
        b = model1.theta.get(key_b)
        if a is None or b is None:
            continue

        hardness = base_hardness
        if tier_fn is not None:
            tier = tier_fn(key_a)
            if tier == "clip-l":
                hardness = hard_l
            elif tier == "clip-g":
                hardness = hard_g
            elif tier in {"t5", "qwen3_4b"}:
                hardness = hard_t5
            elif tier in {"clip", "cap_embedder"}:
                hardness = hard_clip

        merged = _clipxor_semi_hard_blend(
            a, b, hardness=float(hardness), use_cosine_gate=True, keep_stats=True
        )
        if fine:
            merged = _finetune_inplace(key_a, merged, fine, arch=arch or arch_a)
        model0.theta[key_a] = merged
        merged_count += 1

    print(f"[CLIPXOR] pre-merged {merged_count} text-encoder tensors")
    return merged_count


# Mode Functions

def _safe_thread_default() -> int:
    raw = os.environ.get("CHATTIORI_CPU_THREADS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    cpus = os.cpu_count() or 4
    return max(1, min(8, max(1, cpus // 2)))


def _configure_cpu_threads(cpu_threads: int, interop_threads: int) -> None:
    cpu_threads = max(1, int(cpu_threads))
    interop_threads = max(1, int(interop_threads))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(cpu_threads)
    try:
        torch.set_num_threads(cpu_threads)
    except Exception as exc:
        print(f"[resource] torch.set_num_threads({cpu_threads}) failed: {exc}")
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        # PyTorch permits setting this only before inter-op work starts.
        pass
    print(f"[resource] CPU threads={cpu_threads}, interop={interop_threads}")


def main():
    parser = argparse.ArgumentParser(description="Merge two or three models")

    parser.add_argument(
        "mode",
        choices=list(theta_funcs.keys()) + list(_LEGACY_MODE_ALIASES.keys()),
        help="Merging mode. CLIPXOR/XDARE are accepted only as deprecated compatibility aliases.",
    )
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
        "clipxor":          "Apply CLIPXOR to text-encoder tensors before the selected merge mode",
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

    parser.add_argument(
        "--save_precision", "--save-precision",
        dest="save_precision", default="fp32", metavar="PRECISION",
        help=("Output precision. Canonical values: fp32, fp16, bf16, fp8, int8. "
              "Aliases such as full/float32, half/float16, bhalf/bfloat16, "
              "quarter/float8, and i8 are accepted."),
    )
    parser.add_argument(
        "--save-component", "--save_component",
        dest="save_component", action="append", default=[], metavar="COMPONENTS",
        help="Save only the selected component(s). Repeat or use a comma-separated list: unet, vae, clip, clip-l, clip-g, transformer, text, text2, all.",
    )
    parser.add_argument("--seed",   type=int,   help="Random seed for stochastic modes (e.g., DARE)", default=None)
    parser.add_argument("--rebasin",   type=int,   help="ReBasin iterations", default=None)
    parser.add_argument("--vae",    type=str,   help="Path of VAE", default=None, required=False)
    parser.add_argument("--memo",   type=str,   help="Additional info bake in metadata", default=None)
    parser.add_argument("--fine",   type=str,   help="Finetune the given keys on model 0", default=None, required=False)
    parser.add_argument("--fine_sat", type=float, default=1.0,
        help="Direct saturation factor for finetune/tone pass. 1.0=off, 0.75=desaturate 25%%. Works with Anima/AM; also used for VAE conv_out when --vae_sat is 1.0.")
    parser.add_argument("--output",             help="Output file name without extension", default="merged", required=False)
    parser.add_argument("--device", type=str,   help="Device to use, defaults to cpu", default="cpu", required=False)
    parser.add_argument(
        "--cpu-threads", "--cpu_threads", type=int, default=_safe_thread_default(),
        help="Limit PyTorch/BLAS CPU worker threads. Default: CHATTIORI_CPU_THREADS or half the logical CPUs, capped at 8.",
    )
    parser.add_argument(
        "--interop-threads", "--interop_threads", type=int, default=1,
        help="Limit PyTorch inter-op threads. Default: 1.",
    )
    parser.add_argument(
        "--int8-compute-precision", "--int8_compute_precision",
        default=os.environ.get("CHATTIORI_INT8_COMPUTE_DTYPE", "fp16"),
        metavar="PRECISION",
        help="Temporary dtype when loading Chattiori int8 checkpoints: fp16 (default), bf16, or fp32.",
    )
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
        help="Apply RGB saturation scaling inside VAE output (decoder.conv_out). 1.0=off. >1 more saturation, <1 less. Does not require --vae if checkpoint already contains VAE.")

    args = parser.parse_args()
    _configure_cpu_threads(args.cpu_threads, args.interop_threads)
    try:
        configure_save_precision(args)
    except ValueError as exc:
        parser.error(str(exc))

    from Utils import _normalize_int8_compute_dtype
    try:
        _normalize_int8_compute_dtype(args.int8_compute_precision)
    except ValueError as exc:
        parser.error(str(exc))

    requested_mode = str(args.mode)
    if requested_mode in _LEGACY_MODE_ALIASES:
        normalized_mode, enable_clipxor = _LEGACY_MODE_ALIASES[requested_mode]
        print(f"[deprecated] mode {requested_mode} is now {normalized_mode} + --clipxor")
        args.mode = normalized_mode
        args.clipxor = bool(args.clipxor or enable_clipxor)

    try:
        save_components = _parse_save_component_values(args.save_component)
    except ValueError as exc:
        parser.error(str(exc))

    # Keep the original CLI text before args.alpha / args.beta are normalized by wgt(),
    # rand_ratio(), parse_ratio(), or mode-specific code paths.
    _raw_alpha_arg = str(args.alpha)
    _raw_beta_arg = str(args.beta)

    def _json_safe(v):
        """Return a JSON-serializable representation for merge metadata."""
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        if isinstance(v, (np.integer, np.floating, np.bool_)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, torch.Tensor):
            if v.numel() <= 256:
                return v.detach().cpu().tolist()
            return {"tensor_shape": list(v.shape), "dtype": str(v.dtype)}
        if isinstance(v, (list, tuple, set)):
            return [_json_safe(x) for x in v]
        if isinstance(v, OrderedDict):
            return {str(k): _json_safe(val) for k, val in v.items()}
        if isinstance(v, dict):
            return {str(k): _json_safe(val) for k, val in v.items()}
        return str(v)

    def _meta_text(v):
        if v is None:
            return None
        if isinstance(v, str):
            return v if v != "" else None
        return json.dumps(_json_safe(v), ensure_ascii=False, default=str)

    def _path_basename(v: str):
        t = str(v).rstrip("/\\")
        name = os.path.basename(t)
        return name or "<local_path_redacted>"

    def _looks_like_local_path(v: str) -> bool:
        if not isinstance(v, str) or not v:
            return False
        if v.startswith(("http://", "https://", "hf://")):
            return False
        return (
            os.path.isabs(v)
            or v.startswith("~")
            or v.startswith("\\\\")
            or re.match(r"^[A-Za-z]:[\\/]", v) is not None
        )

    def _sanitize_local_paths(v):
        """Remove author-identifying local paths from metadata payloads."""
        if isinstance(v, str):
            return _path_basename(v) if _looks_like_local_path(v) else v
        if isinstance(v, (list, tuple)):
            return [_sanitize_local_paths(x) for x in v]
        if isinstance(v, dict):
            out = {}
            for k, val in v.items():
                lk = str(k).lower()
                if isinstance(val, str) and ("path" in lk or lk in {"file", "filename", "source"}):
                    out[k] = _path_basename(val) if _looks_like_local_path(val) else val
                else:
                    out[k] = _sanitize_local_paths(val)
            return out
        return v

    def _sanitize_info_local_paths(info):
        if info is None:
            return info
        for attr in ("path", "file", "filename", "source", "source_path", "model_path", "cache_path"):
            if hasattr(info, attr):
                try:
                    setattr(info, attr, _sanitize_local_paths(getattr(info, attr)))
                except Exception:
                    pass
        for attr in ("metadata", "extra_info", "parent_models"):
            if hasattr(info, attr):
                try:
                    setattr(info, attr, _sanitize_local_paths(getattr(info, attr)))
                except Exception:
                    pass
        return info

    merge_param_snapshot = None

    def _snapshot_merge_params(
        stage: str,
        *,
        alpha=None,
        beta=None,
        weights_a=None,
        weights_b=None,
        deep_a=None,
        deep_b=None,
        alpha_info=None,
        beta_info=None,
        useblocks=None,
        usebeta=None,
    ):
        nonlocal merge_param_snapshot
        snap = {
            "stage": stage,
            "mode": str(args.mode),
            "effective_mode": str(mode),
            "alpha_raw": _raw_alpha_arg,
            "beta_raw": _raw_beta_arg,
            "alpha": _json_safe(alpha),
            "beta": _json_safe(beta),
            "alpha_info": _meta_text(alpha_info) or _meta_text(alpha) or _raw_alpha_arg,
            "beta_info": _meta_text(beta_info) or (_meta_text(beta) if usebeta else None),
            "alpha_weights": _json_safe(weights_a),
            "beta_weights": _json_safe(weights_b),
            "alpha_deep": _json_safe(deep_a),
            "beta_deep": _json_safe(deep_b),
            "uses_blocks": bool(useblocks) or bool(weights_a is not None) or bool(weights_b is not None),
            "uses_beta": bool(usebeta),
            "rand_alpha": args.rand_alpha,
            "rand_beta": args.rand_beta,
        }
        merge_param_snapshot = _sanitize_local_paths(snap)
        return merge_param_snapshot

    def _fine_sat_active() -> bool:
        return abs(float(getattr(args, "fine_sat", 1.0)) - 1.0) >= 1e-6

    def _with_fine_sat(fine_obj):
        """Attach direct saturation factor to image_tone_v2 fine dict."""
        if not _fine_sat_active():
            return fine_obj
        sat = float(args.fine_sat)
        if isinstance(fine_obj, dict):
            out = dict(fine_obj)
        else:
            out = {
                "mode": "image_tone_v2",
                "noise1": 0.0,
                "noise2": 0.0,
                "noise3": 0.0,
                "contrast": 0.0,
                "brightness": 0.0,
                "r": 0.0,
                "g": 0.0,
                "b": 0.0,
            }
        out["saturation"] = sat
        return out

    needs_model1 = (args.mode not in {"NoIn", "RM", "COMP"}) or bool(args.clipxor)
    if needs_model1 and args.model_1 is None:
        raise SystemExit(f"mode '{args.mode}'{' + CLIPXOR' if args.clipxor else ''} needs model_1")

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
    model0 = model1 = model2 = None

    if mode not in ["SWAP", "COMP"] and not turbo_convert:
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

    cache_path = os.path.join(base_path(), "cache.json")
    merge_cache_json(cache_path, args.model_path)
    output_name, output_file, output_path = _build_output_path(args)

    comp_components = None

    torch.set_grad_enabled(False)

    alpha_seed = beta_seed = None
    alpha_info = beta_info = ""
    if args.rand_alpha is not None:
        args.alpha, alpha_seed, deep_a, alpha_info = rand_ratio(args.rand_alpha)
    if args.rand_beta is not None:
        args.beta, beta_seed,  deep_b,  beta_info  = rand_ratio(args.rand_beta)

    _snapshot_merge_params(
        "initial",
        alpha=args.alpha,
        beta=args.beta,
        deep_a=deep_a,
        deep_b=deep_b,
        alpha_info=alpha_info,
        beta_info=beta_info,
        useblocks=useblocks,
        usebeta=False,
    )

    model_0_path = _model_path(args.model_path, args.model_0)
    model_0_name = args.m0_name or _model_stem(model_0_path)
    print(f"Loading {model_0_name}...")
    model0 = _load_umodel(model_0_path, name=model_0_name, device=device, verify_hash=True, cache_path=cache_path, int8_compute_dtype=args.int8_compute_precision)
    if mode == "RM":
        print(model0.sha256)
        print(json.dumps(model0.metadata, indent=2, ensure_ascii=False))
        with open(f"./{args.output}.json", "a+", encoding="utf-8") as dmp:
            json.dump(model0.metadata, dmp, indent=4, ensure_ascii=False)
        raise SystemExit(0)

    arch = model0.arch

    anima_output_block_count = (
        ANIMA29_BLOCK_COUNT if arch.get("AM29", False) else ANIMA_BASE_BLOCK_COUNT
    ) if arch.get("AM", False) else None

    def _align_anima_source(model, label: str):
        if not arch.get("AM", False) or model is None or not model.has_tensors:
            return model
        if not model.arch.get("AM", False):
            return model
        source_count = detect_anima_block_count(model.theta)
        if source_count == anima_output_block_count:
            return model
        mapped, info = remap_anima_theta(model.theta, anima_output_block_count)
        if info.get("changed"):
            model.theta = mapped
            print(
                f"[Anima compatibility] {label}: {info['source_blocks']} -> "
                f"{info['target_blocks']} main blocks ({info.get('direction', 'remap')})"
            )
        return model

    if arch.get("AM", False):
        variant = "Anima 2.9B (40-block)" if arch.get("AM29", False) else "Anima (28-block)"
        print(f"[model-detect] Base family=Anima; variant={variant}")

    clipxor_tensor_count = 0
    if (not turbo_convert) and (mode not in ["NoIn", "COMP"] or args.clipxor):
        model_1_path = _model_path(args.model_path, args.model_1)
        model_1_name = args.m1_name or _model_stem(model_1_path)
        print(f"Loading {model_1_name}...")
        model1 = _load_umodel(model_1_path, name=model_1_name, device=device, verify_hash=True, cache_path=cache_path, int8_compute_dtype=args.int8_compute_precision)
        _align_anima_source(model1, "model_1")
        if mode == "SWAP":
            model1.theta = normalize_external_text_encoder(model1.theta, arch)
        if (args.fine or _fine_sat_active()) and not arch.get("ZI", False):
            fine = _with_fine_sat(fineman([float(t) for t in args.fine.split(",")], arch) if args.fine else "")
        else:
            fine = ""

        if args.clipxor:
            clipxor_tensor_count = _apply_clipxor_inplace(model0, model1, fine=fine, arch=arch)

        if mode == "NoIn":
            _snapshot_merge_params(
                "CLIPXOR" if args.clipxor else "NoIn",
                alpha=None,
                beta=None,
                weights_a=None,
                weights_b=None,
                deep_a=deep_a,
                deep_b=deep_b,
                alpha_info=None,
                beta_info=None,
                useblocks=False,
                usebeta=False,
            )
            usebeta = False
            weights_a = weights_b = None
            alpha = beta = None

        elif mode == "SWAP":
            components, only = _parse_components_with_only(str(args.alpha))

            if not components:
                components = set(_ALL_COMPONENTS)

            moved, created, skipped, model0.theta = _swap_components_inplace(
                model0.theta, model1.theta,
                components,
                arch,
                src_only=only,
            )
            print(f"[SWAP] components={sorted(list(components))} only={sorted(list(only))}  moved:{moved}  created:{created}  shape_skipped:{skipped}")
            _snapshot_merge_params(
                "SWAP",
                alpha=str(args.alpha),
                beta=None,
                weights_a=None,
                weights_b=None,
                deep_a=deep_a,
                deep_b=deep_b,
                alpha_info=str(args.alpha),
                beta_info=None,
                useblocks=False,
                usebeta=False,
            )

            mode = "NoIn"
            usebeta = False
            weights_a = weights_b = None
            alpha = beta = None

        else:
            weights_a, alpha, alpha_info = parse_ratio(args.alpha, alpha_info, deep_a)
            if mode in modes_need_m2:
                model_2_path = _model_path(args.model_path, args.model_2)
                model_2_name = args.m2_name or _model_stem(model_2_path)
                print(f"Loading {model_2_name}...")
                model2 = _load_umodel(model_2_path, name=model_2_name, device=device, verify_hash=True, cache_path=cache_path, int8_compute_dtype=args.int8_compute_precision)
                _align_anima_source(model2, "model_2")

            usebeta = mode in modes_need_beta
            if usebeta:
                weights_b, beta, beta_info = parse_ratio(args.beta, beta_info, deep_b)
            else:
                weights_b, beta = None, None
            _snapshot_merge_params(
                "merge_ratios_parsed",
                alpha=alpha,
                beta=beta,
                weights_a=weights_a,
                weights_b=weights_b,
                deep_a=deep_a,
                deep_b=deep_b,
                alpha_info=alpha_info,
                beta_info=beta_info,
                useblocks=useblocks,
                usebeta=usebeta,
            )
            if args.rebasin is not None:
                if arch.get("FLUX") or arch.get("ZI") or arch.get("AM") or arch.get("K2"):
                    print("[ReBasin] Unavailable architecture detected, skipping ReBasin (not supported).")
                else:
                    print(f"[ReBasin] Running weight matching (Hungarian)... iter={args.rebasin}")
                    ps = unet_permutation_spec(arch.get("XL", False))
                    perm_01, gain_01 = weight_matching(
                        ps,
                        params_a=model0.theta,
                        params_b=model1.theta,
                        max_iter=args.rebasin,
                        usefp16=True,
                        device=device,
                    )
                    model1.theta = apply_permutation(ps, perm_01, model1.theta)
                    print(f"[ReBasin] (0 <-> 1) average gain: {gain_01:.4f}")

                    if mode in modes_need_m2 and model2 is not None and model2.has_tensors:
                        perm_02, gain_02 = weight_matching(
                            ps,
                            params_a=model0.theta,
                            params_b=model2.theta,
                            max_iter=args.rebasin,
                            usefp16=True,
                            device=device,
                        )
                        model2.theta = apply_permutation(ps, perm_02, model2.theta)
                        print(f"[ReBasin] (0 <-> 2) average gain: {gain_02:.4f}")

    else:
        if turbo_convert:
            # Turbo/de-turbo is a dedicated three-model path.  Older control
            # flow skipped the normal loader but still dereferenced model1/2
            # later, producing AttributeError on valid --turbo/--deturbo runs.
            model_1_path = _model_path(args.model_path, args.model_1)
            model_1_name = args.m1_name or _model_stem(model_1_path)
            print(f"Loading {model_1_name}...")
            model1 = _load_umodel(
                model_1_path, name=model_1_name, device=device, verify_hash=True,
                cache_path=cache_path, int8_compute_dtype=args.int8_compute_precision,
            )
            _align_anima_source(model1, "model_1")
            model_2_path = _model_path(args.model_path, args.model_2)
            model_2_name = args.m2_name or _model_stem(model_2_path)
            print(f"Loading {model_2_name}...")
            model2 = _load_umodel(
                model_2_path, name=model_2_name, device=device, verify_hash=True,
                cache_path=cache_path, int8_compute_dtype=args.int8_compute_precision,
            )
            _align_anima_source(model2, "model_2")
            if (args.fine or _fine_sat_active()) and not arch.get("ZI", False):
                fine = _with_fine_sat(fineman([float(t) for t in args.fine.split(",")], arch) if args.fine else "")
            else:
                fine = ""
            usebeta = False
            weights_a = weights_b = None
            alpha = beta = None
        elif args.mode == "COMP":
            atext = str(args.alpha).strip()
            if atext in {"", "0", "0.0", "none", "None"}:
                atext = "all"
            comp_components = _normalize_components_list(atext)

            if not comp_components:
                comp_components = {"unet", "vae", "clip-l", "clip-g", "clip", "transformer", "text", "text2"}

            before = len(model0.theta)
            model0.theta, kept, total = _filter_state_dict_by_components(model0.theta, comp_components, arch)
            print(f"[COMP] components={sorted(list(comp_components))}  kept:{kept} / {before}")
            _snapshot_merge_params(
                "COMP",
                alpha=sorted(list(comp_components)),
                beta=None,
                weights_a=None,
                weights_b=None,
                deep_a=deep_a,
                deep_b=deep_b,
                alpha_info=str(args.alpha),
                beta_info=None,
                useblocks=False,
                usebeta=False,
            )

            mode = "NoIn"
            model1 = None
            deep_a = deep_b = []
            usebeta = False
            weights_a = weights_b = None
            alpha = beta = None
            # COMP intentionally disables architecture-specific interpolation
            # after filtering because no merge operation remains.
            arch = {t: False for t in arch.keys()}
        else:
            # Plain NoIn: keep model0 architecture information for correct
            # VAE/component/save handling, but disable interpolation state.
            usebeta = False
            weights_a = weights_b = None
            alpha = beta = None
            fine = ""
            _snapshot_merge_params(
                "NoIn", alpha=None, beta=None, weights_a=None, weights_b=None,
                deep_a=deep_a, deep_b=deep_b, alpha_info=None, beta_info=None,
                useblocks=False, usebeta=False,
            )

    if args.vae:
        if args.mode == "COMP" and comp_components is not None and ("vae" not in comp_components):
            print("[COMP] --vae was provided but 'vae' is not selected; skipping VAE bake.")
        else:
            vae_model = _load_umodel(normalize_path(args.vae), device=device, model_type="vae", verify_hash=False, cache_path=cache_path, int8_compute_dtype=args.int8_compute_precision)
            vae_name = vae_model.name or _model_stem(args.vae)


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


    @torch.inference_mode()
    def _cosine_blend_tensor(a: torch.Tensor, b: torch.Tensor, strength: float, keep_stats: bool = True):
        g = _cosine_combined_similarity_tensor(a, b)

        mix = (abs(float(strength)) * g).clamp_(0.0, 1.0)

        out32 = torch.lerp(a.to(torch.float32), b.to(torch.float32), mix)

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

    def _pick_base_and_others_for_cosine(thetaA, thetaB, thetaC, cosine_sel: int):
        if cosine_sel == 0:
            base_idx, base_sd = 0, thetaA
            others = [(thetaB, "alpha"), (thetaC, "beta")]
        elif cosine_sel == 1:
            base_idx, base_sd = 1, thetaB
            others = [(thetaA, "alpha"), (thetaC, "beta")]
        else:
            base_idx, base_sd = 2, thetaC
            others = [(thetaA, "alpha"), (thetaB, "beta")]
        return base_idx, base_sd, others


    def cosine_minmax_grouped(base_dict, other_dict, desc, variant=0, lo=10.0, hi=90.0):
        by_block = {}
        for k in tqdm(base_dict.keys(), desc=desc):
            if "first_stage_model" in k or (("model" not in k and "text_encoders" not in k) and not (arch.get("K2", False) and blockfromkey(k, arch)[0] != "Not Merge")) or k not in other_dict:
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
            model1.theta, model2.theta = maybe_to_qdtype(model1.theta, model2.theta, model1.info.quantization, model2.info.quantization, device)
        diff_inplace(model1.theta, model2.theta, theta_func1, "Getting Difference of Model 1 and 2")
        model2.release_tensors()

    if arch.get("FLUX", False):
        model0.theta, model1.theta = maybe_to_qdtype(model0.theta, model1.theta, model0.info.quantization, model1.info.quantization, device)
        if model2 is not None and model2.has_tensors:
            model0.theta, model2.theta = maybe_to_qdtype(model0.theta, model2.theta, model0.info.quantization, model2.info.quantization, device)

    # if mode == "TS":
    #     model0.theta = clone_dict_tensors(model0.theta)
        
    def _require_live_model(model, label: str, option: str):
        if model is None or not getattr(model, "has_tensors", False):
            raise SystemExit(
                f"{option} requires live tensors for {label}. "
                "The selected merge mode may already have consumed that model; "
                "disable the conflicting difference option or choose a compatible mode."
            )
        return model

    if args.use_dif_21:
        # model2.theta := model1 - model2
        _require_live_model(model1, "model_1", "--use_dif_21")
        _require_live_model(model2, "model_2", "--use_dif_21")
        diff_inplace(model2.theta, model1.theta, get_difference, "Getting Difference of Model 1 and 2")

    if args.use_dif_10:
        # model1.theta := model1 - model0
        _require_live_model(model1, "model_1", "--use_dif_10")
        diff_inplace(model1.theta, model0.theta, get_difference, "Getting Difference of Model 0 and 1")

    if args.use_dif_20:
        # model2.theta := model2 - model0
        _require_live_model(model2, "model_2", "--use_dif_20")
        diff_inplace(model2.theta, model0.theta, get_difference, "Getting Difference of Model 0 and 2")
        

    ZI_WLEN = len(BLOCKIDZI) - 1  # 33
    AM_WLEN = len(anima_blockids_for_arch(arch)) - 1 if arch.get("AM", False) else len(BLOCKIDAM28) - 1
    K2_WLEN = len(BLOCKIDK2) - 1  # 28 layer weights after BASE

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
        if not arch.get("AM", False):
            return _fit_weights_to_len(w, AM_WLEN)
        target_count = ANIMA29_BLOCK_COUNT if arch.get("AM29", False) else ANIMA_BASE_BLOCK_COUNT
        return remap_anima_weights(w, target_count)

    def _fit_weights_for_zi(w):
        return _fit_weights_to_len(w, ZI_WLEN)

    def _fit_weights_for_k2(w):
        return _fit_weights_to_len(w, K2_WLEN)

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
            print("Detected Anima 2.9B (40-block) architecture." if arch.get("AM29", False) else "Detected Anima (28-block) architecture.")
            weights_a = _fit_weights_for_am(weights_a)
            weights_b = _fit_weights_for_am(weights_b) if weights_b is not None else None
        elif arch.get("K2", False) and useblocks:
            print("Detected Krea 2 (K2) architecture.")
            weights_a = _fit_weights_for_k2(weights_a)
            weights_b = _fit_weights_for_k2(weights_b) if weights_b is not None else None
            
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

    _sat_tags_cached = (
        _parse_csv_set(args.sat_boost_tags)
        if args.sat_boost_tags
        else {"IN00", "IN01", "IN02", "IN03", "IN04", "IN05", "M00"}
    )

    def apply_merge_strength_boosts(key: str, cur_a: float, cur_b: float | None, mode: str, vae_key: str):
        # Parsed once per merge instead of once per tensor.
        sat_tags = _sat_tags_cached
            
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
                base_tensor = model0.theta.get(key)
                if isinstance(base_tensor, torch.Tensor) and _is_small_or_norm_or_bias(key, base_tensor):
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
        (anima_blockids_for_arch(arch) if arch.get("AM", False) else
        (BLOCKIDK2 if arch.get("K2", False) else BLOCKID))))
    )
    _TAG2IDX = {t: i for i, t in enumerate(blockids)}

    if cosine_sel is not None:
        vae_key_local = "first_stage_model" if not (arch.get("FLUX", False) or arch.get("ZI", False) or arch.get("K2", False)) else "vae"

        theta_c = model2.theta if (model2 is not None and model2.has_tensors) else None
        base_idx, base_sd, others = _pick_base_and_others_for_cosine(model0.theta, model1.theta, theta_c, cosine_sel)

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

        model0.theta = base_sd
        _snapshot_merge_params(
            "cosine_merge",
            alpha=alpha,
            beta=beta,
            weights_a=weights_a,
            weights_b=weights_b,
            deep_a=deep_a,
            deep_b=deep_b,
            alpha_info=alpha_info,
            beta_info=beta_info,
            useblocks=useblocks,
            usebeta=(beta is not None),
        )

        cosine_applied = True
        mode = "NoIn"
        if model1 is not None:
            model1.release_tensors()
        if model2 is not None:
            model2.release_tensors()
        usebeta = False
        weights_a = weights_b = None
        alpha = beta = None
        deep_a = deep_b = []

    if turbo_convert:
        # ensure refs are loaded
        # B = model_1 (turbo), C = model_2 (base)
        # if deturbo, sign = -1 else +1

        # (load model1.theta, model2.theta as usual; make sure you DO load both)
        # (optional) arch check: detect_arch(model1.theta/model2.theta) matches A

        # parse alpha/weights like usual (default alpha=1.0 recommended for full convert)
        if str(args.alpha).strip() in {"", "0", "0.0"}:
            args.alpha = 1.0

        args.alpha, deep_a, block_a = wgt(args.alpha, [])
        weights_a, alpha, alpha_info = parse_ratio(args.alpha, "", deep_a)

        blockids = (
            BLOCKIDFLUX if arch.get("FLUX", False) else
            (BLOCKIDXLL if arch.get("XL", False) else
            (BLOCKIDZI if arch.get("ZI", False) else
            (anima_blockids_for_arch(arch) if arch.get("AM", False) else
            (BLOCKIDK2 if arch.get("K2", False) else BLOCKID))))
        )
        _TAG2IDX = {t: i for i, t in enumerate(blockids)}

        vae_key = "first_stage_model" if not (arch.get("FLUX", False) or arch.get("ZI", False) or arch.get("K2", False)) else "vae"
        resolver = make_param_resolver(alpha, None, weights_a, None, deep_a, [], blockids, usebeta=False)

        do_fine = bool('fine' in locals() and fine)
        model0.theta = turbo_convert_inplace(
            model0.theta, model1.theta, model2.theta, resolver,
            deturbo=bool(args.deturbo),
            vae_key=vae_key,
            bake_vae_enabled=bake_vae_enabled,
            fine=(fine if do_fine else None),
        )
        _snapshot_merge_params(
            "turbo_convert" if args.turbo else "deturbo_convert",
            alpha=alpha,
            beta=None,
            weights_a=weights_a,
            weights_b=None,
            deep_a=deep_a,
            deep_b=deep_b,
            alpha_info=alpha_info,
            beta_info=None,
            useblocks=(weights_a is not None),
            usebeta=False,
        )

        # stop normal merge path
        mode = "NoIn"
        if model1 is not None:
            model1.release_tensors()
        if model2 is not None:
            model2.release_tensors()
        usebeta = False
        weights_a = weights_b = None
        alpha = beta = None
        deep_a = deep_b = []


    def _is_krea2_transformer_key(key: str) -> bool:
        if not arch.get("K2", False):
            return False
        k = key
        for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        return k.startswith(("first.", "blocks.", "tmlp.", "tproj.", "txtmlp.", "txtfusion.", "last."))

    def _is_mergeable_checkpoint_key(key: str) -> bool:
        return ("model" in key) or ("text_encoders" in key) or _is_krea2_transformer_key(key)

    def remerge_model(target_dict, source_dict, desc, mode, resolver, theta=None):
        t = target_dict
        s = source_dict
        skip = checkpoint_dict_skip_on_merge

        for key in tqdm(s.keys(), desc=desc, total=len(s)):
            if arch.get("FLUX", False):
                continue
            if key in skip:
                continue
            if not _is_mergeable_checkpoint_key(key):
                continue
            if key in t:
                continue

            if mode in {"TRS", "ST"} and theta is not None and key in theta:
                ent = resolver(key)
                if ent is None:
                    t[key] = s[key]
                    continue
                _, _, cur_b = ent
                try:
                    t[key] = torch.lerp(s[key], theta[key], float(cur_b))
                except Exception:
                    t[key] = s[key]
            else:
                t[key] = s[key]
        return t

    if mode not in ["NoIn", "TF"]:
        with torch.inference_mode():
            vae_key = "first_stage_model" if not (arch.get("FLUX", False) or arch.get("ZI", False) or arch.get("K2", False)) else "vae"
            resolver = make_param_resolver(alpha, beta, weights_a, weights_b, deep_a, deep_b, blockids, usebeta)
            
            func = theta_func2
            do_fine = bool('fine' in locals() and fine)
            theta0 = model0.theta
            theta1 = model1.theta if (model1 is not None and model1.has_tensors) else None
            theta2 = model2.theta if (model2 is not None and model2.has_tensors) else None
            skip_keys = checkpoint_dict_skip_on_merge
            for key in tqdm(theta0.keys(), desc=f"{merge_name} Merging...", total=len(theta0)):
                if (not bake_vae_enabled) and (vae_key in key):
                    continue
                if key in skip_keys:
                    continue
                if not _is_mergeable_checkpoint_key(key):
                    continue
                if theta1 is None or (key not in theta1):
                    continue
                if (mode in modes_need_m2) and (usebeta or mode == "TD") and (theta2 is not None) and (key not in theta2):
                    continue

                ent = resolver(key)
                if ent is None:
                    continue
                _, cur_a, cur_b = ent

                a = theta0[key]
                b = theta1[key]
                if (not isinstance(a, torch.Tensor)) or (not isinstance(b, torch.Tensor)):
                    continue
                if (not a.is_floating_point()) or (not b.is_floating_point()):
                    continue
                if a.shape != b.shape:
                    continue

                c = None
                if usebeta and (mode in modes_need_m2) and (usebeta or mode == "TD"):
                    if theta2 is None:
                        continue
                    c = theta2[key]
                    if (not isinstance(c, torch.Tensor)) or (not c.is_floating_point()) or (c.shape != a.shape):
                        continue

                # Keep the legacy 4D channel-overlap path, but compute it before
                # any optional saturation branch so the same tensor views are used.
                if (a.shape != b.shape) and (a.dim() == 4) and (b.dim() == 4) and (a.shape[0] == b.shape[0]) and (a.shape[2:] == b.shape[2:]):
                    use = min(a.shape[1], b.shape[1], 4)
                    ad = a[:, :use, ...]
                else:
                    ad = a

                is_sat_target = (args.sat_profile == "safe_attn2_out" and _is_outblock_attn2_vout_key(key))

                if is_sat_target and (float(args.sat_boost_mix) < 1.0 or float(args.sat_delta_cap_pct) > 0.0):
                    cur_a0, cur_b0 = _clamp_for_mode(mode, float(cur_a), (float(cur_b) if cur_b is not None else None))
                    cur_a1, cur_b1 = apply_merge_strength_boosts(key, cur_a, cur_b, mode=mode, vae_key=vae_key)

                    if usebeta and mode in modes_need_m2:
                        out0 = func(ad, b, c, cur_a0, cur_b0)
                        out1 = func(ad, b, c, cur_a1, cur_b1)
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
                    theta0[key] = _finetune_inplace(key, out, fine, arch=arch) if do_fine else out
                    continue

                cur_a, cur_b = apply_merge_strength_boosts(
                    key, cur_a, cur_b,
                    mode=mode,
                    vae_key=vae_key
                )

                if mode == "sAD":
                    bf = b.detach().float().cpu()
                    filt = scipy.ndimage.gaussian_filter(bf.numpy(), sigma=1)
                    out = a + cur_a * torch.from_numpy(filt).to(a.device, dtype=a.dtype)
                    theta0[key] = _finetune_inplace(key, out, fine, arch=arch) if do_fine else out
                    continue

                if mode == "TD":
                    t1f = b.float()
                    t2f = theta2[key].float()
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
                    theta0[key] = _finetune_inplace(key, out, fine, arch=arch) if do_fine else out
                    continue

                if mode == "TS":
                    if a.dim() == 0:
                        continue
                    n = a.shape[0]
                    if cur_a + cur_b <= 1:
                        s, e = int(n * cur_b), int(n * (cur_a + cur_b))
                        theta0[key][s:e, ...].copy_(b[s:e, ...])
                    else:
                        s, e = int(n * (cur_a + cur_b - 1)), int(n * cur_b)
                        t = b.clone()
                        t[s:e, ...].copy_(a[s:e, ...])
                        theta0[key] = t
                    model0.theta[key] = _finetune_inplace(key, theta0[key], fine, arch=arch) if do_fine else theta0[key]
                    continue

                if (a.shape != b.shape) and (a.dim() == 4) and (b.dim() == 4) and (a.shape[0] == b.shape[0]) and (a.shape[2:] == b.shape[2:]):
                    use = min(a.shape[1], b.shape[1], 4)
                    ad = a[:, :use, ...]
                else:
                    ad = a

                if usebeta and mode in modes_need_m2:
                    out = func(ad, b, c, cur_a, cur_b)
                elif usebeta:
                    out = func(ad, b, cur_a, cur_b)
                else:
                    out = func(ad, b, cur_a)

                theta0[key] = _finetune_inplace(key, out, fine, arch=arch) if do_fine else out

        if mode != "DARE":
            if mode != "AD" and model2 is not None and model2.has_tensors:
                model0.theta = remerge_model(model0.theta, model1.theta, "Remerging...", mode, resolver, theta=model2.theta)
            else:
                model0.theta = remerge_model(model0.theta, model1.theta, "Remerging...", mode, resolver)
        model1.release_tensors()
        try:
            if mode != "AD" and model2 is not None and model2.has_tensors:
                model0.theta = remerge_model(model0.theta, model2.theta, desc="Remerging...", mode=mode, resolver=resolver)
                model2.release_tensors()
        except NameError:
            pass

    else:
        if args.mode == "TF":
            model0.theta = prune_extras_vs_model1(model0.theta, model1.theta)
            resolver = make_param_resolver(alpha, beta, weights_a, weights_b, deep_a, deep_b, blockids, usebeta)
            theta_c = model2.theta if (model2 is not None and model2.has_tensors) else None
            model0.theta = remerge_model(model0.theta, model1.theta, desc="Remerging...", mode=mode, resolver=resolver, theta=theta_c)
        arch, model0.theta = detect_arch(model0.theta)
        vae_key = "first_stage_model" if not (arch.get("FLUX", False) or arch.get("ZI", False) or arch.get("K2", False)) else "vae"
        if (not cosine_applied) and (args.fine or _fine_sat_active()) and not arch.get("ZI", False):
            fine = _with_fine_sat(fineman([float(t) for t in args.fine.split(",")], arch=arch) if args.fine else "")
            for key in tqdm(model0.theta.keys(), desc="Fine Tuning ..."):
                # VAE saturation is handled after optional VAE bake so it also works
                # for embedded Anima/AM VAEs without requiring --vae.
                if args.vae is None and vae_key in key:
                    continue
                model0.theta[key] = _finetune_inplace(key, model0.theta[key], fine, arch=arch)
        elif not cosine_applied:
            fine = ""
            
    def _strip_vae_root(k: str):
        for r in ("vae.", "first_stage_model.", "model.vae.", "model.first_stage_model."):
            if k.startswith(r):
                return k[len(r):]
        return k

    def _is_checkpoint_vae_key(k: str, vae_key: str) -> bool:
        roots = (
            f"{vae_key}.",
            f"model.{vae_key}.",
        )
        return k == vae_key or any(k.startswith(root) for root in roots)

    if args.vae:
        for k in tqdm(vae_model.theta.keys(), desc=f"Baking in VAE[{vae_name}] ..."):
            tk = vae_key + "." + _strip_vae_root(k)
            model0.theta[tk] = to_half(vae_model.theta[k], args.save_half)
        vae_model.release_tensors()
    else:
        vae_key = "first_stage_model" if not (arch.get("FLUX", False) or arch.get("ZI", False) or arch.get("K2", False)) else "vae"
        keep_embedded_vae = bool(save_components and "vae" in save_components)
        if not keep_embedded_vae:
            vae_keys = [k for k in list(model0.theta.keys()) if _is_checkpoint_vae_key(k, vae_key)]
            for k in vae_keys:
                del model0.theta[k]
            if vae_keys:
                print(f"[VAE] --vae not specified; removed {len(vae_keys)} embedded VAE tensors ({vae_key}).")
        else:
            print("[VAE] Keeping embedded VAE because --save-component includes vae.")

    # Apply VAE saturation to the current checkpoint as well, not only when
    # an external --vae is baked. This is the safest direct saturation control
    # for Anima/AM because it mixes the RGB output channels in decoder.conv_out.
    _vae_sat_value = float(args.vae_sat)
    if abs(_vae_sat_value - 1.0) < 1e-6 and _fine_sat_active():
        _vae_sat_value = float(args.fine_sat)
    if abs(_vae_sat_value - 1.0) >= 1e-6:
        model0.theta = apply_vae_saturation_inplace(model0.theta, vae_key=vae_key, sat=_vae_sat_value)

    arch, model0.theta = detect_arch(model0.theta)

    if arch.get("XL", False):
        for k in tqdm([k for k in model0.theta.keys() if "cond_stage_model." in k], desc="Cond resolving..."):
            del model0.theta[k]

    if float(args.cfg_sens) != 1.0:
        if not arch.get("XL", False):
            print("[cfg_sens] Warning: --cfg_sens is tuned for SDXL; applying anyway.")
        model0.theta = apply_cfg_sens_inplace(model0.theta, gain=float(args.cfg_sens), targets=str(args.cfg_sens_targets))

    if save_components:
        before = len(model0.theta)
        model0.theta, kept, total = _filter_state_dict_by_components(model0.theta, save_components, arch)
        print(f"[save-component] components={sorted(save_components)} kept:{kept} / {total}")
        if kept == 0:
            raise SystemExit(
                "--save-component did not match any tensors. Check the component name and model architecture."
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
    if _fine_sat_active():
        calcs.append(f"fine_sat[{args.fine_sat}]")
    if float(args.vae_sat) != 1.0:
        calcs.append(f"vae_sat[{args.vae_sat}]")
    if float(args.cfg_sens) != 1.0:
        calcs.append(f"cfg_sens[{args.cfg_sens}|{args.cfg_sens_targets}]")
    if float(args.sat_boost) != 1.0:
        calcs.append(f"sat_boost[{args.sat_boost}|{args.sat_boost_side}|{args.sat_boost_tags or 'auto'}]")
    if args.clipxor:
        calcs.append(f"clipxor[{clipxor_tensor_count}]")
    if save_components:
        calcs.append("save_component[" + ",".join(sorted(save_components)) + "]")
    calcl = ",".join(calcs) or None

    fp = args.save_precision

    if merge_param_snapshot is None:
        _snapshot_merge_params(
            "final_fallback",
            alpha=locals().get("alpha"),
            beta=locals().get("beta"),
            weights_a=locals().get("weights_a"),
            weights_b=locals().get("weights_b"),
            deep_a=locals().get("deep_a"),
            deep_b=locals().get("deep_b"),
            alpha_info=locals().get("alpha_info"),
            beta_info=locals().get("beta_info"),
            useblocks=locals().get("useblocks"),
            usebeta=locals().get("usebeta"),
        )

    _meta_alpha_info = merge_param_snapshot.get("alpha_info") or (alpha_info or None)
    _meta_beta_info = merge_param_snapshot.get("beta_info") or (beta_info or None)

    merge_recipe = {
        "type":                 "merge-models-chattiori",
        "primary_model_hash":   model0.sha256,
        "secondary_model_hash": model1.sha256 if (model1 is not None) else None,
        "tertiary_model_hash":  model2.sha256 if (model2 is not None) else None,
        "merge_method":         merge_name,
        "requested_mode":       requested_mode,
        "normalized_mode":      args.mode,
        "block_weights":        bool(merge_param_snapshot.get("uses_blocks")),
        "alpha_info":           _meta_alpha_info,
        "beta_info":            _meta_beta_info,
        "alpha_raw":            merge_param_snapshot.get("alpha_raw"),
        "beta_raw":             merge_param_snapshot.get("beta_raw"),
        "alpha":                merge_param_snapshot.get("alpha"),
        "beta":                 merge_param_snapshot.get("beta"),
        "alpha_weights":        merge_param_snapshot.get("alpha_weights"),
        "beta_weights":         merge_param_snapshot.get("beta_weights"),
        "alpha_deep":           merge_param_snapshot.get("alpha_deep"),
        "beta_deep":            merge_param_snapshot.get("beta_deep"),
        "ratio_stage":          merge_param_snapshot.get("stage"),
        "uses_beta":            merge_param_snapshot.get("uses_beta"),
        "calculation":          calcl,
        "fp":                   fp,
        "output_name":          output_name,
        "bake_in_vae":          (vae_name if args.vae else False),
        "pruned":               args.prune,
        "merge_options": {
            "clipxor": bool(args.clipxor),
            "clipxor_tensor_count": int(clipxor_tensor_count),
            "save_component": sorted(save_components),
            "cosine0": bool(args.cosine0),
            "cosine1": bool(args.cosine1),
            "cosine2": bool(args.cosine2),
            "use_dif_10": bool(args.use_dif_10),
            "use_dif_20": bool(args.use_dif_20),
            "use_dif_21": bool(args.use_dif_21),
            "turbo": bool(args.turbo),
            "deturbo": bool(args.deturbo),
            "rebasin": args.rebasin,
            "seed": args.seed,
            "rand_alpha": args.rand_alpha,
            "rand_beta": args.rand_beta,
            "alpha_seed": alpha_seed,
            "beta_seed": beta_seed,
            "fine": args.fine,
            "fine_sat": args.fine_sat,
            "cfg_sens": args.cfg_sens,
            "cfg_sens_targets": args.cfg_sens_targets,
            "sat_boost": args.sat_boost,
            "sat_boost_side": args.sat_boost_side,
            "sat_boost_tags": args.sat_boost_tags,
            "sat_profile": args.sat_profile,
            "sat_delta_cap_pct": args.sat_delta_cap_pct,
            "sat_boost_mix": args.sat_boost_mix,
            "boost_clamp": args.boost_clamp,
            "vae_sat": args.vae_sat,
        },
    }

    if args.clipxor:
        merge_recipe["clipxor"] = {
            "option": True,
            "intersection": "elemwise_minabs_same_sign",
            "base": False,
            "tensor_count": int(clipxor_tensor_count),
        }

    if args.mode == "SWAP":
        merge_recipe["swap_components_alpha_text"] = str(args.alpha)
    elif args.mode == "COMP":
        merge_recipe["comp_components_alpha_text"] = str(args.alpha)
    elif args.rebasin is not None:
        merge_recipe["rebasin"] = {
            "iter": args.rebasin,
            "min_channels": 64,
            "max_channels": 4096,
        }
        
    metadata["sd_merge_recipe"] = json.dumps(_sanitize_local_paths(merge_recipe), ensure_ascii=False, default=str)

    def _coerce_json_dict(v):
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def add_model_metadata(s256, hashed, meta, model_name):
        if s256 is None:
            return

        metadata["sd_merge_models"][s256] = {
            "name": model_name,
            "legacy_hash": hashed,
            "sd_merge_recipe": _coerce_json_dict(meta.get("sd_merge_recipe")),
        }

        prev_models = _coerce_json_dict(meta.get("sd_merge_models"))
        metadata["sd_merge_models"].update(prev_models)

    add_model_metadata(model0.sha256, model0.info.legacy_hash, dict(model0.metadata), model0.name)
    if model1 is not None and model1.sha256 is not None:
        add_model_metadata(model1.sha256, model1.info.legacy_hash, dict(model1.metadata), model1.name)
    if model2 is not None and model2.sha256 is not None:
        add_model_metadata(model2.sha256, model2.info.legacy_hash, dict(model2.metadata), model2.name)

    metadata["sd_merge_models"] = json.dumps(_sanitize_local_paths(metadata["sd_merge_models"]), ensure_ascii=False, default=str)
    metadata = _sanitize_local_paths(metadata)

    delete_targets = []
    if args.delete_source:
        for model_name, loaded_model in [
            (args.model_0, model0),
            (args.model_1, model1),
            (args.model_2, model2),
        ]:
            if model_name is None or loaded_model is None or loaded_model.sha256 is None:
                continue
            p = _model_path(args.model_path, model_name)
            if p and os.path.isfile(p):
                delete_targets.append(p)

    merged_info = _clone_info(model0, name=output_name, path=output_path)
    merged_info.model_type = "checkpoint"
    merged_info.metadata = dict(metadata)
    merged_info.arch = dict(arch)
    # Do not bake author-identifying local paths into unified_model_info.
    # _save_umodel still receives output_path directly, so saving behavior is unchanged.
    try:
        merged_info.path = os.path.basename(output_path)
    except Exception:
        pass
    _sanitize_info_local_paths(merged_info)

    merged = UnifiedModel.from_theta(model0.theta, info=merged_info, clone_tensors=False)
    _register_merge_parents(merged, model0, model1, model2)
    _sanitize_info_local_paths(getattr(merged, "info", None))

    merge_success = False
    try:
        _save_umodel(merged, output_path, args=args)
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

if __name__ == "__main__":
    main()