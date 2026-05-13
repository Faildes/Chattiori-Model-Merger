import os
import re
import json
import argparse
import torch
import torch.nn.functional as F
import safetensors.torch
import safetensors
from tqdm.auto import tqdm
import math

from Utils import (
    LBLOCKS26,
    LBLOCKS_FLUX,
    LBLOCKS_ZI,
    LBLOCKS_SDXL,
    LBLOCKS_AM,
    BLOCKID,
    BLOCKIDFLUX,
    BLOCKIDZI,
    BLOCKIDXLL,
    BLOCKIDAM,
    normalize_path,
    base_path,
    merge_cache_json,
    blockfromkey,
    elementals2,
    _load_umodel,
)

from model import UnifiedModel, ModelInfo

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

_TE2_QKV_DOWN_RE = re.compile(
    r"^lora_te2_text_model_encoder_layers_(\d+)_self_attn_([qkv])_proj\.lora_(?:A|down)\.weight$"
)
_TE2_OUT_DOWN_RE = re.compile(
    r"^lora_te2_text_model_encoder_layers_(\d+)_self_attn_out_proj\.lora_(?:A|down)\.weight$"
)


def resolve_sdxl_te2_target_any(down_k: str, keymap: dict):
    m = _TE2_QKV_DOWN_RE.match(down_k)
    if m:
        li = int(m.group(1))
        part = m.group(2)
        tgt = keymap.get(f"1_model_transformer_resblocks_{li}_attn_in_proj")
        if tgt is not None:
            return tgt, part
        return None, None

    m = _TE2_OUT_DOWN_RE.match(down_k)
    if m:
        li = int(m.group(1))
        tgt = keymap.get(f"1_model_transformer_resblocks_{li}_attn_out_proj")
        if tgt is not None:
            return tgt, None
        return None, None

    return None, None

_DIRECT_LORA_DOWN_SUFFIXES = (
    ".lora_down.weight",
    ".lora_A.weight",
)

def _strip_direct_lora_down_suffix(key: str) -> str | None:
    for suf in _DIRECT_LORA_DOWN_SUFFIXES:
        if key.endswith(suf):
            return key[:-len(suf)]
    return None

def _strip_lora_runtime_suffix(key: str) -> str:
    for suf in (
        ".lora_down.weight",
        ".lora_up.weight",
        ".lora_A.weight",
        ".lora_B.weight",
        ".alpha",
    ):
        if key.endswith(suf):
            return key[:-len(suf)]
    return key

def _normalize_direct_lora_base_for_keymap(base: str) -> str:
    sk = base.replace(".", "_")

    if "conditioner_embedders_" in sk:
        return sk.split("conditioner_embedders_", 1)[1]

    if "text_encoders_" in sk:
        return sk.split("text_encoders_", 1)[1]

    if "wrapped_" in sk:
        return sk.split("wrapped_", 1)[1]

    if sk.startswith("model_"):
        return sk.split("model_", 1)[1]

    return sk

def convert_diffusers_name_to_compvis(key: str, is_sd2: bool) -> str:
    g: list = []
    direct_base = _strip_direct_lora_down_suffix(key)
    if direct_base is not None and not direct_base.startswith((
        "lora_unet_",
        "lora_te_",
        "lora_te1_",
        "lora_te2_",
        "lora_diffusion_model_",
    )):
        return _normalize_direct_lora_base_for_keymap(direct_base)
    
    key = _strip_lora_runtime_suffix(key)
    
    if _m(r"lora_unet_input_blocks_(\d+)_(\d+)_(.+)", key, g):
        return f"diffusion_model_input_blocks_{g[0]}_{g[1]}_{g[2]}"
    if _m(r"lora_unet_middle_block_(\d+)_(.+)", key, g):
        return f"diffusion_model_middle_block_{g[0]}_{g[1]}"
    if _m(r"lora_unet_output_blocks_(\d+)_(\d+)_(.+)", key, g):
        return f"diffusion_model_output_blocks_{g[0]}_{g[1]}_{g[2]}"
    if _m(r"lora_unet_out_(\d+)_(.+)", key, g):
        return f"diffusion_model_out_{g[0]}_{g[1]}"
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

_SDXL_TEXTENC_DIRECT_QKV_RE = re.compile(
    r"^text_encoders\.(clip_l|clip_g)\.transformer\.text_model\.encoder\.layers\.(\d+)\.self_attn\.([qkv])_proj\.lora_(?:A|down)\.weight$"
)
_SDXL_TEXTENC_DIRECT_OUT_RE = re.compile(
    r"^text_encoders\.(clip_l|clip_g)\.transformer\.text_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.lora_(?:A|down)\.weight$"
)
_SDXL_TEXTENC_DIRECT_MLP_RE = re.compile(
    r"^text_encoders\.(clip_l|clip_g)\.transformer\.text_model\.encoder\.layers\.(\d+)\.mlp\.fc([12])\.lora_(?:A|down)\.weight$"
)
_SDXL_TEXTENC_DIRECT_LN_RE = re.compile(
    r"^text_encoders\.(clip_l|clip_g)\.transformer\.text_model\.encoder\.layers\.(\d+)\.layer_norm([12])\.lora_(?:A|down)\.weight$"
)
_SDXL_TEXTENC_DIRECT_POS_RE = re.compile(
    r"^text_encoders\.(clip_l|clip_g)\.transformer\.text_model\.embeddings\.position_embedding\.lora_(?:A|down)\.weight$"
)
_SDXL_TEXTENC_DIRECT_TOKEN_RE = re.compile(
    r"^text_encoders\.(clip_l|clip_g)\.transformer\.text_model\.embeddings\.token_embedding\.lora_(?:A|down)\.weight$"
)
_SDXL_TEXTENC_DIRECT_FINAL_RE = re.compile(
    r"^text_encoders\.(clip_l|clip_g)\.transformer\.text_model\.final_layer_norm\.lora_(?:A|down)\.weight$"
)
_SDXL_TEXTENC_DIRECT_TPROJ_RE = re.compile(
    r"^text_encoders\.(clip_l|clip_g)\.text_projection\.lora_(?:A|down)\.weight$"
)

_SDXL_ADD_EMB_RE = re.compile(r"^lora_unet_add_embedding_linear_(\d+)\.lora_(?:A|down)\.weight$")
_SDXL_UPSAMPLE_RE = re.compile(r"^lora_unet_up_blocks_(\d+)_upsamplers_0_conv\.lora_(?:A|down)\.weight$")

def _first_keymap_hit(keymap: dict, *cands: str):
    for c in cands:
        tgt = keymap.get(c)
        if tgt is not None:
            return tgt
    return None

def resolve_sdxl_special_unet_target_any(down_k: str, keymap: dict):
    """
    returns (target_key, part)
    part is always None here
    """
    m = _SDXL_ADD_EMB_RE.match(down_k)
    if m:
        idx = int(m.group(1))
        if idx == 1:
            # checkpoints differ here, so try multiple canonical candidates
            tgt = _first_keymap_hit(keymap,
                "diffusion_model_label_emb_0_0",
                "diffusion_model_label_emb_0",
                "diffusion_model_add_embedding_linear_1",
            )
            return tgt, None
        if idx == 2:
            tgt = _first_keymap_hit(keymap, 
                "diffusion_model_label_emb_0_2",
                "diffusion_model_label_emb_2",
                "diffusion_model_add_embedding_linear_2",
            )
            return tgt, None

    m = _SDXL_UPSAMPLE_RE.match(down_k)
    if m:
        bi = int(m.group(1))
        # SDXL / checkpoint variants differ in this sub-index, so try both
        tgt = _first_keymap_hit(keymap, 
            f"diffusion_model_output_blocks_{2 + bi*3}_2_conv",
            f"diffusion_model_output_blocks_{2 + bi*3}_1_conv",
            f"diffusion_model_output_blocks_{2 + bi*3}_conv",
        )
        return tgt, None

    return None, None

def resolve_sdxl_textenc_direct_target_any(down_k: str, keymap: dict):
    """
    Support direct diffusers-style SDXL text encoder LoRA keys:
      text_encoders.clip_l.transformer.text_model...
      text_encoders.clip_g.transformer.text_model...
    Returns (target_key, part) where part in {None,'q','k','v'}
    """

    m = _SDXL_TEXTENC_DIRECT_QKV_RE.match(down_k)
    if m:
        enc = m.group(1)
        li = int(m.group(2))
        part = m.group(3)

        if enc == "clip_l":
            tgt = _first_keymap_hit(
                keymap,
                f"clip_l_transformer_text_model_encoder_layers_{li}_self_attn_{part}_proj",
                f"0_transformer_text_model_encoder_layers_{li}_self_attn_{part}_proj",
            )
            return (tgt, None) if tgt is not None else (None, None)

        # clip_g
        tgt = _first_keymap_hit(
            keymap,
            f"clip_g_transformer_text_model_encoder_layers_{li}_self_attn_{part}_proj",
            f"1_model_text_model_encoder_layers_{li}_self_attn_{part}_proj",
        )
        if tgt is not None:
            return tgt, None

        tgt = _first_keymap_hit(
            keymap,
            f"1_model_transformer_resblocks_{li}_attn_in_proj",
        )
        if tgt is not None:
            return tgt, part

        return None, None

    m = _SDXL_TEXTENC_DIRECT_OUT_RE.match(down_k)
    if m:
        enc = m.group(1)
        li = int(m.group(2))

        if enc == "clip_l":
            tgt = _first_keymap_hit(
                keymap,
                f"clip_l_transformer_text_model_encoder_layers_{li}_self_attn_out_proj",
                f"0_transformer_text_model_encoder_layers_{li}_self_attn_out_proj",
            )
            return (tgt, None) if tgt is not None else (None, None)

        tgt = _first_keymap_hit(
            keymap,
            f"clip_g_transformer_text_model_encoder_layers_{li}_self_attn_out_proj",
            f"1_model_text_model_encoder_layers_{li}_self_attn_out_proj",
            f"1_model_transformer_resblocks_{li}_attn_out_proj",
        )
        return (tgt, None) if tgt is not None else (None, None)

    m = _SDXL_TEXTENC_DIRECT_MLP_RE.match(down_k)
    if m:
        enc = m.group(1)
        li = int(m.group(2))
        which = m.group(3)

        if enc == "clip_l":
            tgt = _first_keymap_hit(
                keymap,
                f"clip_l_transformer_text_model_encoder_layers_{li}_mlp_fc{which}",
                f"0_transformer_text_model_encoder_layers_{li}_mlp_fc{which}",
            )
            return (tgt, None) if tgt is not None else (None, None)

        # clip_g
        if which == "1":
            tgt = _first_keymap_hit(
                keymap,
                f"clip_g_transformer_text_model_encoder_layers_{li}_mlp_fc1",
                f"1_model_text_model_encoder_layers_{li}_mlp_fc1",
                f"1_model_transformer_resblocks_{li}_mlp_c_fc",
            )
        else:
            tgt = _first_keymap_hit(
                keymap,
                f"clip_g_transformer_text_model_encoder_layers_{li}_mlp_fc2",
                f"1_model_text_model_encoder_layers_{li}_mlp_fc2",
                f"1_model_transformer_resblocks_{li}_mlp_c_proj",
            )
        return (tgt, None) if tgt is not None else (None, None)

    m = _SDXL_TEXTENC_DIRECT_LN_RE.match(down_k)
    if m:
        enc = m.group(1)
        li = int(m.group(2))
        which = m.group(3)

        if enc == "clip_l":
            tgt = _first_keymap_hit(
                keymap,
                f"clip_l_transformer_text_model_encoder_layers_{li}_layer_norm{which}",
                f"0_transformer_text_model_encoder_layers_{li}_layer_norm{which}",
            )
            return (tgt, None) if tgt is not None else (None, None)

        tgt = _first_keymap_hit(
            keymap,
            f"clip_g_transformer_text_model_encoder_layers_{li}_layer_norm{which}",
            f"1_model_text_model_encoder_layers_{li}_layer_norm{which}",
            f"1_model_transformer_resblocks_{li}_ln_{which}",
        )
        return (tgt, None) if tgt is not None else (None, None)

    m = _SDXL_TEXTENC_DIRECT_POS_RE.match(down_k)
    if m:
        enc = m.group(1)
        if enc == "clip_l":
            tgt = _first_keymap_hit(
                keymap,
                "clip_l_transformer_text_model_embeddings_position_embedding",
                "0_transformer_text_model_embeddings_position_embedding",
            )
            return (tgt, None) if tgt is not None else (None, None)

        tgt = _first_keymap_hit(
            keymap,
            "clip_g_transformer_text_model_embeddings_position_embedding",
            "1_model_text_model_embeddings_position_embedding",
            "1_model_positional_embedding",
        )
        return (tgt, None) if tgt is not None else (None, None)

    m = _SDXL_TEXTENC_DIRECT_TOKEN_RE.match(down_k)
    if m:
        enc = m.group(1)
        if enc == "clip_l":
            tgt = _first_keymap_hit(
                keymap,
                "clip_l_transformer_text_model_embeddings_token_embedding",
                "0_transformer_text_model_embeddings_token_embedding",
            )
            return (tgt, None) if tgt is not None else (None, None)

        tgt = _first_keymap_hit(
            keymap,
            "clip_g_transformer_text_model_embeddings_token_embedding",
            "1_model_text_model_embeddings_token_embedding",
            "1_model_token_embedding",
        )
        return (tgt, None) if tgt is not None else (None, None)

    m = _SDXL_TEXTENC_DIRECT_FINAL_RE.match(down_k)
    if m:
        enc = m.group(1)
        if enc == "clip_l":
            tgt = _first_keymap_hit(
                keymap,
                "clip_l_transformer_text_model_final_layer_norm",
                "0_transformer_text_model_final_layer_norm",
            )
            return (tgt, None) if tgt is not None else (None, None)

        tgt = _first_keymap_hit(
            keymap,
            "clip_g_transformer_text_model_final_layer_norm",
            "1_model_text_model_final_layer_norm",
            "1_model_ln_final",
        )
        return (tgt, None) if tgt is not None else (None, None)

    m = _SDXL_TEXTENC_DIRECT_TPROJ_RE.match(down_k)
    if m:
        enc = m.group(1)
        if enc == "clip_l":
            tgt = _first_keymap_hit(
                keymap,
                "clip_l_text_projection",
                "0_text_projection",
            )
            return (tgt, None) if tgt is not None else (None, None)

        tgt = _first_keymap_hit(
            keymap,
            "clip_g_text_projection",
            "1_model_text_projection",
        )
        return (tgt, None) if tgt is not None else (None, None)

    return None, None

_AM_UNET_BASE_RE = re.compile(r"^lora_unet_blocks_(\d+)_(.+)$")
_AM_UNET_DOT_BASE_RE = re.compile(
    r"^(?:(?:model\.)?diffusion_model\.|net\.)?blocks\.(\d+)\.(.+)$"
)
_AM_FINAL_LAYER_DOT_BASE_RE = re.compile(
    r"^(?:(?:model\.)?diffusion_model\.|net\.)?final_layer\.(.+)$"
)
_AM_LLM_ADAPTER_DOT_BASE_RE = re.compile(
    r"^(?:(?:model\.)?diffusion_model\.|net\.)?llm_adapter\.blocks\.(\d+)\.(.+)$"
)
_AM_LLM_ADAPTER_ROOT_DOT_BASE_RE = re.compile(
    r"^(?:(?:model\.)?diffusion_model\.|net\.)?llm_adapter\.(?!blocks\.)(.+)$"
)
_AM_T_EMBEDDER_DOT_BASE_RE = re.compile(
    r"^(?:(?:model\.)?diffusion_model\.|net\.)?t_embedder\.(.+)$"
)
_AM_X_EMBEDDER_DOT_BASE_RE = re.compile(
    r"^(?:(?:model\.)?diffusion_model\.|net\.)?x_embedder\.(.+)$"
)
_AM_TE_BASE_RE   = re.compile(r"^lora_te_layers_(\d+)_(.+)$")
_AM_VAE_BASE_RE  = re.compile(r"^(?:lora_vae|lora_first_stage_model)_(.+)$")

def _am_strip_lora_suffix(down_k: str) -> str | None:
    if down_k.endswith(".lora_down.weight"):
        return down_k[:-len(".lora_down.weight")]
    if down_k.endswith(".lora_A.weight"):
        return down_k[:-len(".lora_A.weight")]
    return None

def _am_merge_pairs(tokens: list[str]) -> list[str]:
    # q_proj, k_proj, v_proj, o_proj, out_proj, output_proj, *_norm >> single token
    out = []
    i = 0
    pair2 = {"proj", "norm"}
    head_ok = {"q","k","v","o","out","output"}
    while i < len(tokens):
        if (i + 1) < len(tokens) and (tokens[i] in head_ok) and (tokens[i+1] in pair2):
            out.append(tokens[i] + "_" + tokens[i+1])
            i += 2
            continue
        out.append(tokens[i])
        i += 1
    return out

def _am_tail_to_path(tail: str) -> list[str]:
    """
    Convert Anima kohya-style LoRA tail names to likely checkpoint module paths.

    Examples:
      self_attn_q_proj              -> self_attn.q_proj
      cross_attn_output_proj        -> cross_attn.output_proj / cross_attn.o_proj / cross_attn.out_proj
      mlp_layer1                    -> mlp.layer1
      lora_te_layers_*_mlp_gate_proj -> mlp.gate_proj
    """
    tail = str(tail or "").strip().strip("_")
    if not tail:
        return []

    # Explicit aliases for the exact Anima key families seen in CR.txt / GS.txt.
    explicit: dict[str, list[str]] = {}
    for attn in ("self_attn", "cross_attn"):
        explicit[f"{attn}_q_proj"] = [f"{attn}.q_proj", f"{attn}.to_q"]
        explicit[f"{attn}_k_proj"] = [f"{attn}.k_proj", f"{attn}.to_k"]
        explicit[f"{attn}_v_proj"] = [f"{attn}.v_proj", f"{attn}.to_v"]
        explicit[f"{attn}_o_proj"] = [
            f"{attn}.o_proj", f"{attn}.output_proj", f"{attn}.out_proj", f"{attn}.to_out.0", f"{attn}.to_out"
        ]
        explicit[f"{attn}_out_proj"] = [
            f"{attn}.out_proj", f"{attn}.o_proj", f"{attn}.output_proj", f"{attn}.to_out.0", f"{attn}.to_out"
        ]
        explicit[f"{attn}_output_proj"] = [
            f"{attn}.output_proj", f"{attn}.o_proj", f"{attn}.out_proj", f"{attn}.to_out.0", f"{attn}.to_out"
        ]

    explicit.update({
        "mlp_layer1": ["mlp.layer1", "mlp.fc1", "mlp.w1", "mlp.gate_proj"],
        "mlp_layer2": ["mlp.layer2", "mlp.fc2", "mlp.w2", "mlp.down_proj"],
        "mlp_gate_proj": ["mlp.gate_proj", "mlp.w1", "mlp.layer1"],
        "mlp_up_proj":   ["mlp.up_proj", "mlp.w3", "mlp.layer1"],
        "mlp_down_proj": ["mlp.down_proj", "mlp.w2", "mlp.layer2"],
        "input_layernorm": ["input_layernorm"],
        "post_attention_layernorm": ["post_attention_layernorm"],
        "final_layernorm": ["final_layernorm"],
    })

    paths = list(explicit.get(tail, []))

    groups = [
        "adaln_modulation_cross_attn",
        "adaln_modulation_self_attn",
        "adaln_modulation_mlp",
        "cross_attn",
        "self_attn",
        "mlp",
    ]
    grp = None
    rest = None
    for g in sorted(groups, key=len, reverse=True):
        if tail.startswith(g + "_"):
            grp = g
            rest = tail[len(g) + 1:]
            break

    if grp is None:
        paths.append(tail.replace("_", "."))
    elif rest and rest.isdigit():
        paths.append(f"{grp}.{rest}")
    elif rest:
        toks = _am_merge_pairs(rest.split("_"))
        sub = ".".join(toks)
        paths.append(f"{grp}.{sub}")

        if sub.endswith("o_proj"):
            base = sub[:-len("o_proj")]
            paths.extend([f"{grp}.{base}output_proj", f"{grp}.{base}out_proj"])
        if sub.endswith("output_proj"):
            base = sub[:-len("output_proj")]
            paths.extend([f"{grp}.{base}o_proj", f"{grp}.{base}out_proj"])
        if sub.endswith("out_proj"):
            base = sub[:-len("out_proj")]
            paths.extend([f"{grp}.{base}o_proj", f"{grp}.{base}output_proj"])

    # Delete duplicates while preserving order.
    seen = set()
    out = []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _am_existing_or_first(theta_0: dict, candidates: list[str]) -> str | None:
    for cand in candidates:
        if cand in theta_0:
            return cand
    return candidates[0] if candidates else None


def _am_weight_candidates(prefixes: list[str], idx: int, paths: list[str]) -> list[str]:
    out = []
    for pref in prefixes:
        for path in paths:
            out.append(f"{pref}{idx}.{path}.weight")
    return out


def _am_prefixed_weight_candidates(prefixes: list[str], paths: list[str]) -> list[str]:
    out = []
    for pref in prefixes:
        for path in paths:
            out.append(f"{pref}{path}.weight")
    return out


def _am_expand_numeric_mlp_tail(tail: str) -> list[str]:
    """Add common aliases for direct MLP numeric modules such as mlp.0 / mlp.2."""
    tail = str(tail or "").strip().strip(".")
    if tail == "mlp.0":
        return ["mlp.0", "mlp.layer1", "mlp.fc1", "mlp.gate_proj"]
    if tail == "mlp.2":
        return ["mlp.2", "mlp.layer2", "mlp.fc2", "mlp.down_proj"]
    if tail == "mlp.1":
        return ["mlp.1", "mlp.up_proj"]
    return [tail] if tail else []


def _am_dot_tail_to_paths(tail: str) -> list[str]:
    """
    Convert direct dotted Anima LoRA tails to checkpoint module paths.

    Examples from AB-style LoRAs:
      cross_attn.k_proj                  -> cross_attn.k_proj
      mlp.layer1                         -> mlp.layer1
      adaln_modulation_self_attn.1       -> adaln_modulation_self_attn.1
    """
    tail = str(tail or "").strip().strip(".")
    if not tail:
        return []

    paths = _am_expand_numeric_mlp_tail(tail)

    # Some Anima checkpoints/trainers differ only in output projection naming.
    # Handle both nested tails (cross_attn.o_proj) and root tails (out_proj).
    if tail in {"o_proj", "out_proj", "output_proj"}:
        paths.extend(["o_proj", "out_proj", "output_proj"])
    if tail.endswith(".o_proj"):
        root = tail[:-len(".o_proj")]
        paths.extend([root + ".output_proj", root + ".out_proj"])
    if tail.endswith(".output_proj"):
        root = tail[:-len(".output_proj")]
        paths.extend([root + ".o_proj", root + ".out_proj"])
    if tail.endswith(".out_proj"):
        root = tail[:-len(".out_proj")]
        paths.extend([root + ".o_proj", root + ".output_proj"])

    # Direct files may include a shortened diffusion root in the LoRA key,
    # but the actual checkpoint usually stores model.diffusion_model.*.
    seen = set()
    out = []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def anima_target_candidates_from_lora_down(down_k: str) -> list[str]:
    """Return candidate checkpoint weight keys for an Anima LoRA down key."""
    base = _am_strip_lora_suffix(down_k)
    if base is None:
        return []

    # --- DiT side: kohya-style Anima keys ---
    #   lora_unet_blocks_0_cross_attn_k_proj.lora_A.weight
    m = _AM_UNET_BASE_RE.match(base)
    if m:
        bi = int(m.group(1))
        tail = m.group(2)
        paths = _am_tail_to_path(tail)
        return _am_weight_candidates([
            "model.diffusion_model.blocks.",
            "net.blocks.",
            "blocks.",
            "diffusion_model.blocks.",
        ], bi, paths)

    # --- DiT side: direct module-path Anima keys ---
    #   diffusion_model.blocks.0.cross_attn.k_proj.lora_A.weight
    #   model.diffusion_model.blocks.0.adaln_modulation_mlp.1.lora_down.weight
    m = _AM_UNET_DOT_BASE_RE.match(base)
    if m:
        bi = int(m.group(1))
        tail = m.group(2)
        paths = _am_dot_tail_to_paths(tail)
        return _am_weight_candidates([
            "model.diffusion_model.blocks.",
            "net.blocks.",
            "blocks.",
            "diffusion_model.blocks.",
        ], bi, paths)

    # --- DiT final layer: direct module-path Anima keys ---
    #   diffusion_model.final_layer.linear.lora_down.weight
    #   diffusion_model.final_layer.adaln_modulation.1.lora_down.weight
    m = _AM_FINAL_LAYER_DOT_BASE_RE.match(base)
    if m:
        tail = m.group(1)
        paths = _am_dot_tail_to_paths(tail)
        return _am_prefixed_weight_candidates([
            "model.diffusion_model.final_layer.",
            "net.final_layer.",
            "final_layer.",
            "diffusion_model.final_layer.",
        ], paths)

    # --- LLM adapter blocks: direct module-path Anima keys ---
    #   diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.lora_down.weight
    #   diffusion_model.llm_adapter.blocks.0.mlp.0.lora_down.weight
    m = _AM_LLM_ADAPTER_DOT_BASE_RE.match(base)
    if m:
        bi = int(m.group(1))
        tail = m.group(2)
        paths = _am_dot_tail_to_paths(tail)
        return _am_weight_candidates([
            "model.diffusion_model.llm_adapter.blocks.",
            "net.llm_adapter.blocks.",
            "llm_adapter.blocks.",
            "diffusion_model.llm_adapter.blocks.",
        ], bi, paths)

    # --- LLM adapter root: direct module-path Anima keys ---
    #   diffusion_model.llm_adapter.embed.lora_down.weight
    #   diffusion_model.llm_adapter.out_proj.lora_down.weight
    m = _AM_LLM_ADAPTER_ROOT_DOT_BASE_RE.match(base)
    if m:
        tail = m.group(1)
        paths = _am_dot_tail_to_paths(tail)
        return _am_prefixed_weight_candidates([
            "model.diffusion_model.llm_adapter.",
            "net.llm_adapter.",
            "llm_adapter.",
            "diffusion_model.llm_adapter.",
        ], paths)

    # --- Time embedder: direct module-path Anima keys ---
    #   diffusion_model.t_embedder.1.linear_1.lora_down.weight
    #   diffusion_model.t_embedder.1.linear_2.lora_down.weight
    m = _AM_T_EMBEDDER_DOT_BASE_RE.match(base)
    if m:
        tail = m.group(1)
        paths = _am_dot_tail_to_paths(tail)
        return _am_prefixed_weight_candidates([
            "model.diffusion_model.t_embedder.",
            "net.t_embedder.",
            "t_embedder.",
            "diffusion_model.t_embedder.",
        ], paths)

    # --- Input embedder: direct module-path Anima keys ---
    #   diffusion_model.x_embedder.proj.1.lora_down.weight
    m = _AM_X_EMBEDDER_DOT_BASE_RE.match(base)
    if m:
        tail = m.group(1)
        paths = _am_dot_tail_to_paths(tail)
        return _am_prefixed_weight_candidates([
            "model.diffusion_model.x_embedder.",
            "net.x_embedder.",
            "x_embedder.",
            "diffusion_model.x_embedder.",
        ], paths)

    # --- Text encoder side (Qwen3-0.6B) ---
    m = _AM_TE_BASE_RE.match(base)
    if m:
        li = int(m.group(1))
        tail = m.group(2)
        paths = _am_tail_to_path(tail)
        return _am_weight_candidates([
            "cond_stage_model.qwen3_06b.transformer.model.layers.",
            "cond_stage_model.qwen3_06b_base.transformer.model.layers.",
            "cond_stage_model.qwen3_06b.model.layers.",
            "cond_stage_model.qwen3_06b_base.model.layers.",
            "text_encoders.qwen3_06b.transformer.model.layers.",
            "text_encoders.qwen3_06b_base.transformer.model.layers.",
            "text_encoders.qwen3_06b.model.layers.",
            "text_encoders.qwen3_06b_base.model.layers.",
            "qwen3_06b.transformer.model.layers.",
            "qwen3_06b_base.transformer.model.layers.",
            "qwen3_06b.model.layers.",
            "qwen3_06b_base.model.layers.",
        ], li, paths)

    # --- VAE side (optional) ---
    mv = _AM_VAE_BASE_RE.match(base)
    if mv:
        tail = mv.group(1)
        dotted = tail.replace("_", ".")
        return [
            f"first_stage_model.{dotted}.weight",
            f"vae.{dotted}.weight",
            f"model.first_stage_model.{dotted}.weight",
        ]

    return []


def anima_resolve_target_any(down_k: str, theta_0: dict) -> str | None:
    """AM LoRA down key -> existing checkpoint weight key when possible."""
    return _am_existing_or_first(theta_0, anima_target_candidates_from_lora_down(down_k))


_ZI_LAYER_RE = re.compile(
    r"^(?:model\.)?diffusion_model\.layers\.(\d+)\.(.+)\.lora_(A|down)\.weight$"
)

_ZI_DOWN_SPLIT_RE = re.compile(r"^(.*)\.lora_(?:A|down)\.weight$")
_ZI_DOT_BASE_RE   = re.compile(r"^(?:model\.)?diffusion_model\.layers\.(\d+)\.(.+)$")
_ZI_US_BASE_RE    = re.compile(r"^(?:lora_(?:unet|diffusion_model)_)?layers_(\d+)_(.+)$")
_ZI_US2_BASE_RE   = re.compile(r"^diffusion_model_layers_(\d+)_(.+)$")

_ZI_NORM_RE = re.compile(r"[^a-z0-9]+")

def _zi_norm_tail(s: str) -> str:
    s = s.lower()
    s = s.replace("weight", "")
    return _ZI_NORM_RE.sub("", s)

def _zi_extract_layer_tail_from_downkey(down_k: str):
    m = _ZI_DOWN_SPLIT_RE.match(down_k)
    if not m:
        return None, None
    base = m.group(1)

    m2 = _ZI_DOT_BASE_RE.match(base)
    if m2:
        return int(m2.group(1)), m2.group(2)

    m3 = _ZI_US_BASE_RE.match(base)
    if m3:
        return int(m3.group(1)), m3.group(2)

    m4 = _ZI_US2_BASE_RE.match(base)
    if m4:
        return int(m4.group(1)), m4.group(2)

    return None, None

def _build_zi_layer_index(theta_0: dict):
    """
    layer -> { normalized_tail -> full_weight_key }
    weight key is: model.diffusion_model.layers.{i}.{tail}.weight
    """
    idx = {}
    prefix = "model.diffusion_model.layers."
    for k in theta_0.keys():
        if not (k.startswith(prefix) and k.endswith(".weight")):
            continue
        rest = k[len(prefix):]  # "{i}.{tail}.weight"
        p = rest.find(".")
        if p < 0:
            continue
        li = rest[:p]
        if not li.isdigit():
            continue
        layer = int(li)
        tail = rest[p + 1 : -len(".weight")]
        idx.setdefault(layer, {})[_zi_norm_tail(tail)] = k
    return idx

def zimage_resolve_target_any(down_k: str, zi_index: dict):
    layer, tail = _zi_extract_layer_tail_from_downkey(down_k)
    if layer is None:
        return None, None

    m = zi_index.get(layer)
    if not m:
        return None, None

    t = tail
    tu = re.sub(r"[./]", "_", t).lower()

    if "attention" in tu and "to_q" in tu:
        k_sep = m.get(_zi_norm_tail("attention.to_q")) or m.get(_zi_norm_tail("attention_to_q"))
        if k_sep:
            return k_sep, None
        k_fused = m.get(_zi_norm_tail("attention.qkv")) or m.get(_zi_norm_tail("attention_qkv"))
        if k_fused:
            return k_fused, "q"
        return None, None

    if "attention" in tu and "to_k" in tu:
        k_sep = m.get(_zi_norm_tail("attention.to_k")) or m.get(_zi_norm_tail("attention_to_k"))
        if k_sep:
            return k_sep, None
        k_fused = m.get(_zi_norm_tail("attention.qkv")) or m.get(_zi_norm_tail("attention_qkv"))
        if k_fused:
            return k_fused, "k"
        return None, None

    if "attention" in tu and "to_v" in tu:
        k_sep = m.get(_zi_norm_tail("attention.to_v")) or m.get(_zi_norm_tail("attention_to_v"))
        if k_sep:
            return k_sep, None
        k_fused = m.get(_zi_norm_tail("attention.qkv")) or m.get(_zi_norm_tail("attention_qkv"))
        if k_fused:
            return k_fused, "v"
        return None, None

    if "attention" in tu and ("to_out" in tu or tu.endswith("_out")):
        for cand in ("attention.out", "attention.to_out.0", "attention.to_out", "attention_out", "attention_to_out_0"):
            kk = m.get(_zi_norm_tail(cand))
            if kk:
                return kk, None

    kk = m.get(_zi_norm_tail(t))
    if kk:
        return kk, None
    kk = m.get(_zi_norm_tail(tu))
    if kk:
        return kk, None

    return None, None

def _am_target_key_from_lora_down(down_k: str) -> str | None:
    cands = anima_target_candidates_from_lora_down(down_k)
    return cands[0] if cands else None

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
def apply_zimage_lora_to_weight(
    W: torch.Tensor,
    part: str | None,
    up: torch.Tensor,
    down: torch.Tensor,
    alpha,
    ratio: float,
) -> torch.Tensor | None:
    if W is None or (not isinstance(W, torch.Tensor)):
        return None

    rank = int(down.size(0))
    a = alpha
    if isinstance(a, torch.Tensor):
        a = float(a.item())
    a = float(rank if a is None else a)

    scale = (a / float(rank))
    alpha_mm = float(ratio) * scale

    orig_dtype = W.dtype
    compute_dtype = torch.float32 if (W.device.type == "cpu") else orig_dtype

    Wc = W.to(dtype=compute_dtype)
    uc = up.to(device=W.device, dtype=compute_dtype)
    dc = down.to(device=W.device, dtype=compute_dtype)

    if Wc.ndim != 2:
        apply_lora_to_weight_inplace(Wc, uc, dc, scale, ratio=float(ratio))
    else:
        if part is None:
            Wc.addmm_(uc, dc, beta=1.0, alpha=alpha_mm)
        else:
            d = int(uc.shape[0])
            off = {"q": 0, "k": d, "v": 2 * d}[part]

            if Wc.shape[0] >= off + d and Wc.shape[0] in (3 * d, off + d, Wc.shape[0]):
                view = Wc.narrow(0, off, d)
                view.addmm_(uc, dc, beta=1.0, alpha=alpha_mm)
            elif Wc.shape[1] >= off + d and Wc.shape[1] in (3 * d, off + d, Wc.shape[1]):
                view = Wc.narrow(1, off, d)
                view.addmm_(uc, dc, beta=1.0, alpha=alpha_mm)
            else:
                return None

    if compute_dtype != orig_dtype:
        Wc = Wc.to(orig_dtype)

    return Wc


def _load_lora(path: str, *, device: str = "cpu", cache_path: str | None = None) -> tuple[UnifiedModel, dict, bool]:
    lm = _load_umodel(path, device=device, model_type="lora", verify_hash=True, cache_path=cache_path)
    meta = dict(lm.metadata)
    keys = list(lm.theta.keys())
    isv2 = any("resblocks" in k for k in keys)
    return lm, meta, isv2


def _make_checkpoint_model(theta: dict[str, torch.Tensor], base: UnifiedModel, *, output: str, metadata: dict) -> UnifiedModel:
    info = base.info.clone()
    info.path = output
    info.name = os.path.splitext(os.path.basename(output))[0]
    info.model_type = "checkpoint"
    info.metadata = dict(metadata)
    info.arch = dict(base.arch)
    out = UnifiedModel.from_theta(theta, info=info, clone_tensors=False)
    out.register_parent(base, role="base_checkpoint")
    return out


def _make_lora_model(theta: dict[str, torch.Tensor], *, output: str, metadata: dict, arch: dict | None = None) -> UnifiedModel:
    info = ModelInfo(
        path=output,
        name=os.path.splitext(os.path.basename(output))[0],
        format="safetensors",
        model_type="lora",
        metadata=dict(metadata),
        arch=dict(arch or {}),
    )
    return UnifiedModel.from_theta(theta, info=info, clone_tensors=False)


def _save_umodel(model: UnifiedModel, output: str, *, args) -> None:
    model.save(
        output,
        no_metadata=bool(args.no_metadata),
        save_half=bool(getattr(args, "save_half", False)),
        save_quarter=bool(getattr(args, "save_quarter", False)),
        save_bhalf=bool(getattr(args, "save_bhalf", False)),
        prune=bool(getattr(args, "prune", False)),
        args=args,
    )


def _load_state_dict_umodel(path: str, dtype=torch.float, device="cpu", depatch=True, *, model_type: str | None = None):
    um = _load_umodel(path, device=device, model_type=model_type, verify_hash=False)
    sd = um.theta
    if depatch:
        for k, v in list(sd.items()):
            if isinstance(v, torch.Tensor):
                sd[k] = v.to(dtype=dtype, device=device)
    isv2 = any("resblocks" in k for k in sd.keys())
    return um, sd, dict(um.metadata), isv2


def load_state_dict(path: str, dtype=torch.float, device="cpu", depatch=True, *, model_type: str | None = None):
    _um, sd, meta, isv2 = _load_state_dict_umodel(path, dtype=dtype, device=device, depatch=depatch, model_type=model_type)
    return sd, meta, isv2

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
def build_apply_plan(main_keys, *, arch: dict, mlv2: bool, keymap: dict, theta_0: dict | None = None):
    plan = []
    for k in main_keys:
        down_k, up_k, alpha_k = parse_lora_key(k)
        if down_k is None:
            continue

        if arch.get("XL", False):
            tgt, part = resolve_sdxl_textenc_direct_target_any(down_k, keymap)
            if tgt is not None:
                kind = "fused" if part in ("q", "k", "v") else "std"
                plan.append((kind, down_k, up_k, alpha_k, tgt, part))
                continue

            tgt, part = resolve_sdxl_te2_target_any(down_k, keymap)
            if tgt is not None:
                plan.append(("fused", down_k, up_k, alpha_k, tgt, part))
                continue

        if arch.get("ZI", False):
            tgt, part = zimage_resolve_target(down_k)
            if tgt is not None:
                plan.append(("zi", down_k, up_k, alpha_k, tgt, part))
            continue

        if arch.get("AM", False):
            if theta_0 is None:
                continue
            tgt = anima_resolve_target_any(down_k, theta_0)
            if tgt is None:
                continue
            plan.append(("std", down_k, up_k, alpha_k, tgt, None))
            continue

        full = convert_diffusers_name_to_compvis(down_k, mlv2)
        msd  = full.split(".", 1)[0]
        if arch.get("XL", False):
            msd = msd.replace("lora_unet", "diffusion_model").replace("lora_te1_text_model", "0_transformer_text_model")

        wkey = keymap.get(msd)
        if wkey is None:
            continue
        plan.append(("std", down_k, up_k, alpha_k, wkey, None))
    return plan

def _split_top_level(s: str, sep: str = ",") -> list[str]:
    # split by sep, but ignore seps inside (), [], {}, and quotes
    opens = {"(": ")", "[": "]", "{": "}"}
    stack = []
    out = []
    buf = []
    quote = None
    escape = False
    for ch in s:
        if quote is not None:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue

        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            continue
        if ch in opens:
            stack.append(opens[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
        elif (ch == sep) and (not stack):
            part = "".join(buf).strip()
            if part:
                out.append(part)
            buf = []
            continue
        buf.append(ch)
    part = "".join(buf).strip()
    if part:
        out.append(part)
    return out

_WINDOWS_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_UNC_PATH_RE = re.compile(r"^(?:\\\\|//)[^\\/]+[\\/][^\\/]+")
_LORA_FILE_EXTS = (
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
)

def _strip_outer_quotes(s: str) -> str:
    s = str(s or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1].strip()
    return s

def _is_windows_drive_colon(s: str, i: int) -> bool:
    return (
        i == 1
        and len(s) >= 3
        and s[0].isalpha()
        and s[1] == ":"
        and s[2] in ("/", "\\")
    )

def _is_absish_path(path: str) -> bool:
    p = _strip_outer_quotes(path)
    return bool(os.path.isabs(p) or _WINDOWS_DRIVE_PATH_RE.match(p) or _UNC_PATH_RE.match(p))

def _has_lora_file_ext(path: str) -> bool:
    return _strip_outer_quotes(path).lower().endswith(_LORA_FILE_EXTS)

def _resolve_lora_path(path: str, model_path: str) -> str:
    """
    Resolve LoRA path while preserving absolute Windows paths such as X:/... and X:\\...
    even when this script is executed from a non-Windows-compatible pathlib/os.path context.
    """
    p = _strip_outer_quotes(path)
    if _is_absish_path(p):
        return normalize_path(p)
    return normalize_path(os.path.join(model_path, p))

def _top_level_colon_positions(s: str) -> list[int]:
    opens = {"(": ")", "[": "]", "{": "}"}
    stack = []
    positions = []
    quote = None
    escape = False
    for i, ch in enumerate(s):
        if quote is not None:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue

        if ch in ("'", '"'):
            quote = ch
            continue
        if ch in opens:
            stack.append(opens[ch])
            continue
        if stack and ch == stack[-1]:
            stack.pop()
            continue
        if ch == ":" and not stack:
            if _is_windows_drive_colon(s, i):
                continue
            positions.append(i)
    return positions

def _candidate_lora_path_exists(path: str, model_path: str | None) -> bool:
    p = _strip_outer_quotes(path)
    candidates = []
    if _is_absish_path(p):
        candidates.append(normalize_path(p))
    elif model_path:
        candidates.append(normalize_path(os.path.join(model_path, p)))
    else:
        candidates.append(normalize_path(p))

    expanded = []
    for c in candidates:
        expanded.append(c)
        if not os.path.splitext(c)[1]:
            expanded.extend(c + ext for ext in _LORA_FILE_EXTS)
    return any(os.path.exists(c) for c in expanded)

def _split_lora_spec(item: str, *, model_path: str | None = None) -> tuple[str, str]:
    """
    Split one LoRA spec into (path, ratio) without confusing:
      - Windows drive prefixes: X:/foo/bar.safetensors
      - Elemental/deep ratio expressions that contain ':'

    Preferred delimiter is the first top-level ':' immediately after a known LoRA
    file extension, e.g. X:/a/b.safetensors:[IN04:attn:0.5].
    If no extension boundary is present, fall back to an existing resolved path,
    then finally to the first non-drive top-level colon for legacy name:ratio.
    """
    s = _strip_outer_quotes(item)
    if not s:
        return "", "1.0"

    positions = _top_level_colon_positions(s)
    if not positions:
        return s.strip(), "1.0"

    # Strongest signal: path with a LoRA/checkpoint extension before delimiter.
    for pos in positions:
        left = s[:pos].strip()
        right = s[pos + 1:].strip()
        if left and right and _has_lora_file_ext(left):
            return _strip_outer_quotes(left), right

    # Next: choose the delimiter where left resolves to an existing file.
    for pos in positions:
        left = s[:pos].strip()
        right = s[pos + 1:].strip()
        if left and right and _candidate_lora_path_exists(left, model_path):
            return _strip_outer_quotes(left), right

    # Legacy fallback: name:ratio.  This intentionally keeps additional ':' in ratio.
    pos = positions[0]
    left = s[:pos].strip()
    right = s[pos + 1:].strip()
    return _strip_outer_quotes(left), (right or "1.0")

def get_loralist(arg: str, model_path: str | None = None):
    parts = _split_top_level(str(arg or ""), ",")
    out = []
    for x in parts:
        p, r = _split_lora_spec(x, model_path=model_path)
        if p:
            out.append([p.strip(), (r or "1.0").strip()])
    return out

def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        return s[1:-1].strip()
    return s

def _split_rules(s: str) -> list[str]:
    if not s:
        return []
    buf = s.strip()
    if (buf.startswith("[") and buf.endswith("]")) or (buf.startswith("{") and buf.endswith("}")):
        buf = buf[1:-1].strip()

    parts = _split_top_level(buf, ",")
    if len(parts) == 1:
        parts = []
        for x in re.split(r"[;\n]+", buf):
            x = x.strip()
            if x:
                parts.append(x)

    out = []
    for p in parts:
        p = _strip_quotes(p.strip())
        if p:
            out.append(p)
    return out

def _parse_float_list(s: str) -> list[float]:
    t = s.strip()
    if t.startswith("[") and t.endswith("]"):
        t = t[1:-1].strip()
    if t == "":
        return []
    xs = [x.strip() for x in _split_top_level(t, ",")]
    out = []
    for x in xs:
        x = _strip_quotes(x)
        if x == "":
            continue
        out.append(float(x))
    return out

def _consume_one_bracket_group(s: str, open_ch: str = "[", close_ch: str = "]"):
    if not s.startswith(open_ch):
        return None, s
    depth = 0
    for i, ch in enumerate(s):
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return s[: i + 1], s[i + 1 :].lstrip()
    return None, s

def parse_ratio_spec(ratio_str: str, nblocks: int):
    """
    backward compatible + elementals
      "1.0"                         -> g=1.0, weights=[1.0], deep=[]
      "0.8,0.8,0.8"                 -> g=1.0, weights=[0.8,0.8,0.8], deep=[]
      "[...]"                       -> g=1.0, weights=[...], deep=[]
      "g,[weights]"                 -> g=g, weights=[...], deep=[]
      "g,[[weights],[deep...]]"     -> g=g, weights=[...], deep=[...]
      "[weights][deep...]"          -> g=1.0, weights=[...], deep=[...]
      "{deep...}" / "[deep...]"     -> g=1.0, weights=[1.0], deep=[...]
    """
    s = (ratio_str or "").strip()
    if s == "":
        return 1.0, [1.0], []

    g = 1.0
    parts = _split_top_level(s, ",")
    if len(parts) >= 2 and _is_float(parts[0]) and parts[1].lstrip().startswith(("[", "{")):
        g = float(parts[0])
        rest = s[len(parts[0]):].lstrip()
        if rest.startswith(","):
            rest = rest[1:].lstrip()
    else:
        rest = s

    weights: list[float] = []
    deep: list[str] = []

    if rest.lstrip().startswith("["):
        rest = rest.lstrip()
        g1, tail = _consume_one_bracket_group(rest, "[", "]")
        if g1 is not None:
            tail2 = tail.lstrip()
            g2 = None
            if tail2.startswith("["):
                g2, tail3 = _consume_one_bracket_group(tail2, "[", "]")
            elif tail2.startswith("{"):
                g2, tail3 = _consume_one_bracket_group(tail2, "{", "}")

            inner = g1[1:-1].strip()
            inner_parts = _split_top_level(inner, ",")
            if (len(inner_parts) >= 2) and inner_parts[0].lstrip().startswith("[") and inner_parts[1].lstrip().startswith("["):
                weights = _parse_float_list(inner_parts[0])
                deep = _split_rules(inner_parts[1])
            else:
                try:
                    weights = _parse_float_list(g1)
                    if len(weights) == 0:
                        deep = _split_rules(g1)
                        weights = [1.0]
                except Exception:
                    deep = _split_rules(g1)
                    weights = [1.0]

            if g2 is not None:
                deep2 = _split_rules(g2)
                if deep2:
                    deep.extend(deep2)

            if not weights:
                weights = [1.0]

            return g, weights, deep

    if rest.lstrip().startswith("{") and rest.rstrip().endswith("}"):
        deep = _split_rules(rest)
        return g, [1.0], deep

    try:
        weights = _parse_float_list(rest)
        if not weights:
            weights = [float(rest)]
    except Exception:
        weights = [1.0]

    return g, weights, deep

def _weight_index_for_target_key(target_key: str, blockids: list[str], *, arch: dict) -> int:
    left, right = blockfromkey(target_key, arch=arch)

    if right in blockids:
        return blockids.index(right)
    if left in blockids:
        return blockids.index(left)

    return 0

def _effective_ratio_for_target(
    target_key: str,
    *,
    g: float,
    weights: list[float],
    deep: list[str],
    blockids: list[str],
    arch: dict,
) -> float:
    wi = _weight_index_for_target_key(target_key, blockids, arch=arch)
    base = weights[wi] if (weights and wi < len(weights)) else (weights[0] if weights else 1.0)
    r0 = float(g) * float(base)

    r = elementals2(
        target_key, wi, deep, r0,
        blockids=blockids,
        arch=arch,
    )
    return float(r)

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

        elif "text_encoders_" in sk:
            parts = sk.split("text_encoders_", 1)
            if len(parts) == 2:
                km[parts[1]] = k

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


def _is_text_encoder_lora_key(down_k: str, target_key: str | None = None) -> bool:
    s = down_k
    if target_key:
        s = s + " " + target_key
    s = s.lower()
    # kohya/diffusers/flux-ish
    return (
        "lora_te" in s or
        "text_model" in s or
        "text_encoder" in s or
        "text_encoders" in s or
        "transformer_text_model" in s or
        "conditioner" in s or
        "clip" in s
    )

@torch.inference_mode()
def _preprocess_up_down(
    up: torch.Tensor,
    down: torch.Tensor,
    *,
    rank_cap: int = 0,
    clamp_q: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # to 2d
    up2d, down2d, is_conv, shape_info = _to_2d_up_down(up, down)

    # rank cap
    if rank_cap and rank_cap > 0 and int(up2d.shape[1]) > int(rank_cap):
        up2d, down2d = _compress_uv_rankcap_dimstyle(up2d, down2d, int(rank_cap))

    # clamp
    if clamp_q and clamp_q > 0.0:
        up2d, down2d = _clamp_uv_quantile_inplace(up2d, down2d, float(clamp_q))

    # reshape back compatible with apply_lora_to_weight_inplace
    if is_conv:
        in_ch, kh, kw = shape_info
        r = int(down2d.shape[0])
        down_new = down2d.reshape(r, int(in_ch), int(kh), int(kw))
        up_new   = up2d.reshape(int(up2d.shape[0]), r, 1, 1)
    else:
        down_new = down2d
        up_new   = up2d

    return up_new, down_new

def _apply_delta_cap_to_ratio(
    W: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    *,
    sc: float,
    ratio: float,
    cap: float,
) -> float:
    """
    Enforce ||ΔW||_F <= cap * ||W||_F (approx upper bound using ||U||_F||V||_F).
    ΔW ≈ (ratio*sc) * (up@down)
    """
    if not cap or cap <= 0.0:
        return ratio

    # compute in float32
    Wn = torch.linalg.norm(W.detach().float()).item()
    if not (Wn > 0.0):
        return ratio

    up2d, down2d, _is_conv, _shape = _to_2d_up_down(up, down)
    Un = torch.linalg.norm(up2d.float()).item()
    Vn = torch.linalg.norm(down2d.float()).item()
    if not (Un > 0.0 and Vn > 0.0):
        return ratio

    # upper bound of ||up@down||_F <= ||up||_F||down||_F
    denom = abs(float(ratio) * float(sc)) * Un * Vn + 1e-12
    limit = float(cap) * Wn
    if denom <= limit:
        return ratio

    factor = limit / denom
    return float(ratio) * float(factor)

def _estimate_rel_update_bound(
    Wn0: float,
    up: torch.Tensor,
    down: torch.Tensor,
    *,
    sc: float,
    ratio: float,
) -> float:
    """
    Estimate rho = ||ΔW||_F / ||W0||_F by upper bound:
      ||ΔW||_F <= |ratio*sc| * ||U||_F * ||V||_F
    where U=up2d, V=down2d
    """
    if not (Wn0 > 0.0):
        return 0.0

    up2d, down2d, _is_conv, _shape = _to_2d_up_down(up, down)
    Un = torch.linalg.norm(up2d.float()).item()
    Vn = torch.linalg.norm(down2d.float()).item()
    if not (Un > 0.0 and Vn > 0.0):
        return 0.0

    num = abs(float(ratio) * float(sc)) * float(Un) * float(Vn)
    return float(num) / (float(Wn0) + 1e-12)

def _parse_budget_spec(s: str, nblocks: int):
    """
    returns ("off", None) or ("zones", [vals...]) or ("ratio", (g, weights, deep))
    zones:
      "a:b" or "a:b:c" with floats, applied by block index split
    ratio:
      reuse parse_ratio_spec (e.g. "0.18", "0.18,[...]", "[...]" etc)
    """
    if not s:
        return ("off", None)
    t = s.strip()
    if (":" in t) and ("[" not in t) and ("{" not in t):
        parts = [p.strip() for p in t.split(":")]
        if parts and all(_is_float(p) for p in parts):
            vals = [float(p) for p in parts]
            if len(vals) in (2, 3):
                return ("zones", vals)

    g, weights, deep = parse_ratio_spec(t, nblocks)
    return ("ratio", (float(g), list(weights), list(deep)))


def _budget_for_target(
    target_key: str,
    *,
    spec,
    blockids: list[str],
    arch: dict,
) -> float:
    kind, payload = spec
    if kind == "off" or payload is None:
        return 0.0

    wi = _weight_index_for_target_key(target_key, blockids, arch=arch)
    n = max(1, len(blockids))

    if kind == "zones":
        vals = payload
        if len(vals) == 2:
            cut = n // 2
            return float(vals[0] if wi < cut else vals[1])
        else:
            cut1 = n // 3
            cut2 = (2 * n) // 3
            if wi < cut1:   return float(vals[0])
            if wi < cut2:   return float(vals[1])
            return float(vals[2])

    # ratio-like
    g, weights, deep = payload
    return float(_effective_ratio_for_target(
        target_key,
        g=g, weights=weights, deep=deep,
        blockids=blockids,
        arch=arch,
    ))

def _estimate_rel_delta_bound(
    Wn: float,
    up: torch.Tensor,
    down: torch.Tensor,
    *,
    sc: float,
    ratio: float,
) -> float:
    """
    Estimate rho = ||ΔW||_F / ||W||_F using upper bound:
      ||ΔW||_F <= |ratio*sc| * ||U||_F ||V||_F
    """
    if not (Wn > 0.0):
        return float("inf")

    up2d, down2d, _is_conv, _shape = _to_2d_up_down(up, down)
    Un = torch.linalg.norm(up2d.float()).item()
    Vn = torch.linalg.norm(down2d.float()).item()
    if not (Un > 0.0 and Vn > 0.0):
        return 0.0

    denom = abs(float(ratio) * float(sc)) * float(Un) * float(Vn)
    return float(denom) / (float(Wn) + 1e-12)


@torch.inference_mode()
def pluslora(lora_list, model, output, model_path, device="cpu"):
    cache_path = os.path.join(base_path(), "cache.json")
    merge_cache_json(cache_path, model_path)
    model_path = normalize_path(model_path)
    if not model:     return "ERROR: No model Selected"
    if not lora_list: return "ERROR: No LoRA Selected"

    print("Plus LoRA start")

    # checkpoint
    mpath = normalize_path(os.path.join(model_path, model))
    base_model = _load_umodel(mpath, device=device, model_type="checkpoint", verify_hash=True, cache_path=cache_path)

    zi_index = None
    if base_model.arch.get("ZI", False):
        zi_index = _build_zi_layer_index(base_model.theta)
        
    Wn_cache = {}
    for k, W in base_model.theta.items():
        if isinstance(W, torch.Tensor) and W.is_floating_point():
            Wn_cache[k] = float(torch.linalg.norm(W.detach().float()).item())
        else:
            Wn_cache[k] = 0.0

    def _get_Wn(key: str) -> float:
        return float(Wn_cache.get(key, 0.0))
        
    # ---- W0 norms (fixed) for order-independent budget ----
    W0_norm: dict[str, float] = {}
    for k, W in base_model.theta.items():
        if isinstance(W, torch.Tensor) and W.is_floating_point():
            W0_norm[k] = float(torch.linalg.norm(W.detach().float()).item())
        else:
            W0_norm[k] = 0.0

    def _get_W0n(key: str) -> float:
        return float(W0_norm.get(key, 0.0))

    blocks = (
        LBLOCKS_ZI if base_model.arch.get("ZI", False) else
        (LBLOCKS_FLUX if base_model.arch.get("FLUX", False) else
        (LBLOCKS_SDXL if base_model.arch.get("XL", False) else
        (LBLOCKS_AM if base_model.arch.get("AM", False) else LBLOCKS26)))
    )

    blocknum = (
        BLOCKIDZI if base_model.arch.get("ZI", False) else
        (BLOCKIDFLUX if base_model.arch.get("FLUX", False) else
        (BLOCKIDXLL if base_model.arch.get("XL", False) else
        (BLOCKIDAM if base_model.arch.get("AM", False) else BLOCKID)))
    )

    keymap   = _build_keymap(base_model.theta)

    lr_strs   = []
    lora_meta = {}
    
    # ---- global normalization when baking many LoRAs ----
    n_loras = max(1, len(lora_list))
    if getattr(args, "bake_norm", "sqrt") == "mean":
        bake_norm = 1.0 / float(n_loras)
    elif getattr(args, "bake_norm", "sqrt") == "sqrt":
        bake_norm = 1.0 / math.sqrt(float(n_loras))
    else:
        bake_norm = 1.0
    bake_norm *= float(getattr(args, "bake_scale", 1.0))

    bake_unet_only = bool(getattr(args, "bake_unet_only", False))
    bake_rank_cap  = int(getattr(args, "bake_rank_cap", 0))
    bake_clamp_q   = float(getattr(args, "bake_clamp_q", 0.0))
    bake_delta_cap = float(getattr(args, "bake_delta_cap", 0.0))
    bake_clip_scale = float(getattr(args, "bake_clip_scale", 1.0))
    bake_delta_cap_user = float(getattr(args, "bake_delta_cap", 0.0))
    bake_guard = str(getattr(args, "bake_guard", "auto"))
    bake_guard_cap = float(getattr(args, "bake_guard_cap", 0.05))
    bake_guard_skip = float(getattr(args, "bake_guard_skip", 0.25))
    budget_spec = _parse_budget_spec(str(getattr(args, "bake_budget", "")).strip(), len(blocknum))
    use_budget = (budget_spec[0] != "off")

    # effective delta cap
    bake_delta_cap_eff = bake_delta_cap_user
    if bake_guard == "cap":
        bake_delta_cap_eff = bake_guard_cap
    elif bake_guard == "auto":
        if (n_loras >= 2) and (bake_delta_cap_user <= 0.0):
            bake_delta_cap_eff = bake_guard_cap
            
    bake_fp32      = bool(getattr(args, "bake_fp32", False))
    
    from collections import defaultdict
    sum_rho = defaultdict(float)
    
    if use_budget:
        print("Budget pass-1: collecting per-target update sums...")

        for lora_model, ratio_str in lora_list:
            lora_path = _resolve_lora_path(lora_model, model_path)
            lora_model_obj, meta, lisv2 = _load_lora(lora_path, device=device, cache_path=cache_path)

            g, weights, deep = parse_ratio_spec(ratio_str, len(blocknum))

            for down_k in tqdm(list(_iter_lora_down_keys(lora_model_obj.theta)), desc=f"Budget scan {lora_model}...", leave=False):
                bake_norm_use = bake_norm
                d, u, a = _pair_from_down_key(down_k)
                if d is None or (u not in lora_model_obj.theta) or (d not in lora_model_obj.theta):
                    continue
                if bake_unet_only and _is_text_encoder_lora_key(d):
                    continue
                
                if _is_text_encoder_lora_key(d):
                    bake_norm_use *= bake_clip_scale

                direct_te_target, direct_te_part = (None, None)
                te2_target, te2_part = (None, None)
                special_target, special_part = (None, None)

                if base_model.arch.get("XL", False):
                    direct_te_target, direct_te_part = resolve_sdxl_textenc_direct_target_any(d, keymap)
                    if direct_te_target is None:
                        te2_target, te2_part = resolve_sdxl_te2_target_any(d, keymap)
                        special_target, special_part = resolve_sdxl_special_unet_target_any(d, keymap)

                if direct_te_target is not None:
                    target_key = direct_te_target
                elif te2_target is not None:
                    target_key = te2_target
                elif special_target is not None:
                    target_key = special_target
                elif base_model.arch.get("AM", False):
                    target_key = anima_resolve_target_any(d, base_model.theta)
                    if target_key is None or (target_key not in base_model.theta):
                        continue
                else:
                    full = convert_diffusers_name_to_compvis_cached(d, bool(lisv2))
                    msd  = full.split(".", 1)[0]
                    if base_model.arch.get("XL", False):
                        msd = msd.replace("lora_unet","diffusion_model").replace("lora_te1_text_model","0_transformer_text_model")
                    target_key = keymap.get(msd)
                    if target_key is None or (target_key not in base_model.theta):
                        continue

                # ratio (include bake_norm)
                ratio_eff = _effective_ratio_for_target(
                    target_key,
                    g=g, weights=weights, deep=deep,
                    blockids=blocknum,
                    arch=base_model.arch,
                )
                ratio_eff = float(ratio_eff) * float(bake_norm_use)

                down = lora_model_obj.theta[d]
                up   = lora_model_obj.theta[u]
                if not (isinstance(down, torch.Tensor) and isinstance(up, torch.Tensor)):
                    continue
                dim = int(down.size(0))
                alpha = lora_model_obj.theta.get(a, dim)
                if alpha is None and isinstance(a, str) and a.endswith(".weight"):
                    alpha = lora_model_obj.theta.get(a[:-len(".weight")], dim)
                if isinstance(alpha, torch.Tensor):
                    alpha = float(alpha.item())
                sc = float(alpha) / float(dim)

                # preprocess (same as apply side) to keep estimate consistent
                up_p, down_p = _preprocess_up_down(up, down, rank_cap=bake_rank_cap, clamp_q=bake_clamp_q)

                Wn0 = _get_W0n(target_key)
                rho = _estimate_rel_update_bound(Wn0, up_p, down_p, sc=sc, ratio=ratio_eff)
                if rho > 0.0:
                    sum_rho[target_key] += float(rho)

            del lora_model_obj.theta

    for lora_model, ratio_str in lora_list:
        print(f"loading: {lora_model}")

        g, weights, deep = parse_ratio_spec(ratio_str, len(blocknum))
        lr_strs.append(ratio_str)

        lora_path = _resolve_lora_path(lora_model, model_path)
        lora_model_obj, meta, lisv2 = _load_lora(lora_path, device=device, cache_path=cache_path)
        lora_meta[lora_model_obj.sha256] = meta

        # --- apply plan build ---
        plan = []  # (kind, target_key, part, down_k, up_k, alpha_k, ratio)
        unresolved = 0
        unresolved_examples = []
        for down_k in tqdm(list(_iter_lora_down_keys(lora_model_obj.theta)), desc=f"Planning {lora_model}..."):
            bake_norm_use = bake_norm
            d, u, a = _pair_from_down_key(down_k)
            if d is None:
                continue
            if (u not in lora_model_obj.theta) or (d not in lora_model_obj.theta):
                continue
            if bake_unet_only and _is_text_encoder_lora_key(d):
                continue
            if _is_text_encoder_lora_key(d):
                bake_norm_use *= bake_clip_scale
            
            if base_model.arch.get("ZI", False):
                target_key, part = zimage_resolve_target_any(d, zi_index)
                if target_key is None:
                    continue
                ratio = _effective_ratio_for_target(
                    target_key,
                    g=g, weights=weights, deep=deep,
                    blockids=blocknum,
                    arch=base_model.arch,
                )
                plan.append(("zi", target_key, part, d, u, a, float(ratio)))
                continue
            
            if base_model.arch.get("XL", False):
                direct_te_target, direct_te_part = resolve_sdxl_textenc_direct_target_any(d, keymap)
                if direct_te_target is not None:
                    ratio = _effective_ratio_for_target(
                        direct_te_target,
                        g=g, weights=weights, deep=deep,
                        blockids=blocknum,
                        arch=base_model.arch,
                    )
                    ratio = float(ratio) * float(bake_norm_use)
                    kind = "fused" if direct_te_part in ("q", "k", "v") else "std"
                    plan.append((kind, direct_te_target, direct_te_part, d, u, a, float(ratio)))
                    continue

                te2_target, te2_part = resolve_sdxl_te2_target_any(d, keymap)
                if te2_target is not None:
                    ratio = _effective_ratio_for_target(
                        te2_target,
                        g=g, weights=weights, deep=deep,
                        blockids=blocknum,
                        arch=base_model.arch,
                    )
                    ratio = float(ratio) * float(bake_norm_use)
                    plan.append(("fused", te2_target, te2_part, d, u, a, float(ratio)))
                    continue
                
                special_target, special_part = resolve_sdxl_special_unet_target_any(d, keymap)
                if special_target is not None:
                    ratio = _effective_ratio_for_target(
                        special_target,
                        g=g, weights=weights, deep=deep,
                        blockids=blocknum,
                        arch=base_model.arch,
                    )
                    ratio = float(ratio) * float(bake_norm_use)
                    plan.append(("std", special_target, special_part, d, u, a, float(ratio)))
                    continue

            if base_model.arch.get("AM", False):
                target = anima_resolve_target_any(d, base_model.theta)
                if target is None or target not in base_model.theta:
                    unresolved += 1
                    if len(unresolved_examples) < 12:
                        unresolved_examples.append(d)
                    continue

                ratio = _effective_ratio_for_target(
                    target,
                    g=g, weights=weights, deep=deep,
                    blockids=blocknum,
                    arch=base_model.arch,
                )
                ratio = float(ratio) * float(bake_norm_use)
                plan.append(("std", target, None, d, u, a, float(ratio)))
                continue

            # standard (SD/SDXL/Flux)
            full = convert_diffusers_name_to_compvis_cached(d, lisv2)
            msd  = full.split(".", 1)[0]
            if base_model.arch.get("XL", False):
                msd = msd.replace("lora_unet","diffusion_model").replace("lora_te1_text_model","0_transformer_text_model")

            target = keymap.get(msd)
            if target is None:
                unresolved += 1
                if len(unresolved_examples) < 12:
                    unresolved_examples.append(d)
                continue

            ratio = _effective_ratio_for_target(
                target,
                g=g, weights=weights, deep=deep,
                blockids=blocknum,
                arch=base_model.arch,
            )
            ratio = float(ratio) * float(bake_norm_use)
            plan.append(("std", target, None, d, u, a, float(ratio)))

        print(f"[plan] {lora_model}: resolved={len(plan)} unresolved={unresolved}")
        if unresolved_examples:
            print("[plan] unresolved samples:")
            for s in unresolved_examples:
                print("  -", s)

        # --- apply ---
        for kind, target_key, part, down_k, up_k, alpha_k, ratio in tqdm(plan, desc=f"Merging {lora_model}...", leave=False):
            if target_key not in base_model.theta:
                continue

            # alpha / scale
            down = lora_model_obj.theta[down_k]
            up   = lora_model_obj.theta[up_k]
            dim  = int(down.size(0))
            alpha = lora_model_obj.theta.get(alpha_k, dim)
            if alpha is None and isinstance(alpha_k, str) and alpha_k.endswith(".weight"):
                alpha = lora_model_obj.theta.get(alpha_k[:-len(".weight")], dim)

            sc = float(alpha) / float(dim)
            
            # preprocess (rank cap + clamp) on CPU tensors
            up_p, down_p = _preprocess_up_down(
                up, down, rank_cap=bake_rank_cap, clamp_q=bake_clamp_q
            )

            # ---- Guard: auto scale down / skip only problematic modules ----
            if bake_guard != "none":
                Wn = _get_Wn(target_key)
                rho = _estimate_rel_delta_bound(Wn, up_p, down_p, sc=sc, ratio=ratio)

                # too extreme -> skip this module only
                if (bake_guard_skip and bake_guard_skip > 0.0) and (rho > bake_guard_skip):
                    # (optional) log
                    # print(f"[guard-skip] {lora_model} :: {target_key} rho={rho:.3f} ratio={ratio:.4g}")
                    continue

                # cap -> scale ratio down to satisfy cap
                if (bake_delta_cap_eff and bake_delta_cap_eff > 0.0) and (rho > bake_delta_cap_eff):
                    ratio = float(ratio) * float(bake_delta_cap_eff / max(rho, 1e-12))
                    # (optional) log
                    # print(f"[guard-cap ] {lora_model} :: {target_key} rho={rho:.3f}->{bake_delta_cap_eff:.3f} ratio-> {ratio:.4g}")

            # delta cap (adjust ratio downward if too strong for this module)
            if bake_delta_cap > 0.0:
                # for zi we cap vs the actual target weight
                W_ref = base_model.theta.get(target_key)
                if isinstance(W_ref, torch.Tensor):
                    ratio = _apply_delta_cap_to_ratio(W_ref, up_p, down_p, sc=sc, ratio=ratio, cap=bake_delta_cap)

            # overwrite local up/down used below
            up, down = up_p, down_p
            
            if kind == "fused":
                W = base_model.theta.get(target_key)
                if not isinstance(W, torch.Tensor):
                    continue

                out = apply_zimage_lora_to_weight(
                    W, part=part,
                    up=up, down=down,
                    alpha=alpha,
                    ratio=ratio,
                )
                if out is not None:
                    base_model.theta[target_key] = out
                continue

            if kind == "zi":
                W = base_model.theta.get(target_key)
                if not isinstance(W, torch.Tensor):
                    continue

                out = apply_zimage_lora_to_weight(
                    W, part=part,
                    up=up, down=down,
                    alpha=alpha,
                    ratio=ratio,
                )
                if out is not None:
                    base_model.theta[target_key] = out
                continue

            W = base_model.theta[target_key]
            dev = W.device

            if bake_fp32 and isinstance(W, torch.Tensor) and W.is_floating_point():
                W_work = W.float()
                u = up.to(device=dev, dtype=torch.float32, non_blocking=False)
                d = down.to(device=dev, dtype=torch.float32, non_blocking=False)
                W_out = apply_lora_to_weight_inplace(W_work, u, d, sc, ratio)
                base_model.theta[target_key] = W_out.to(dtype=W.dtype)
            else:
                dt = W.dtype if (isinstance(W, torch.Tensor) and W.is_floating_point()) else torch.float32
                u = up.to(device=dev, dtype=dt, non_blocking=False)
                d = down.to(device=dev, dtype=dt, non_blocking=False)
                base_model.theta[target_key] = apply_lora_to_weight_inplace(W, u, d, sc, ratio)
            
        del lora_model_obj.theta

    out_name = os.path.splitext(os.path.basename(output))[0]
    meta_new = {
        "sd_merge_models": json.dumps({
            "type": "pluslora-chattiori",
            "checkpoint_hash": base_model.sha256 or "",
            "lora_hash": ",".join([k for k in lora_meta.keys() if k]),
            "alpha_info": ",".join(lr_strs),
            "output_name": out_name,
        }),
        "checkpoint": json.dumps(base_model.metadata),
        "lora": json.dumps(lora_meta),
    }
    if args.memo is not None:
        meta_new["memo"] = args.memo

    baked = _make_checkpoint_model(base_model.theta, base_model, output=output, metadata=meta_new)
    print(f"Saving as {output}...")
    _save_umodel(baked, output, args=args)
    del base_model, baked
    print(f"Done! ({round(os.path.getsize(output)/1073741824, 2)}G)")


@torch.inference_mode()
def darelora(mainlora, lora_list, model, output, model_path, device="cpu"):
    cache_path = os.path.join(base_path(), "cache.json")
    model_path = normalize_path(model_path)
    if not model:
        return "ERROR: No model Selected"
    if not lora_list:
        return "ERROR: No LoRA Selected"

    print("Plus LoRA DARE start")

    mpath = normalize_path(os.path.join(model_path, model))
    base_model = _load_umodel(
        mpath,
        device=device,
        model_type="checkpoint",
        verify_hash=True,
        cache_path=cache_path,
    )

    zi_index = None
    if base_model.arch.get("ZI", False):
        zi_index = _build_zi_layer_index(base_model.theta)

    blocks = (
        LBLOCKS_ZI if base_model.arch.get("ZI", False) else
        (LBLOCKS_FLUX if base_model.arch.get("FLUX", False) else
         (LBLOCKS_SDXL if base_model.arch.get("XL", False) else
          (LBLOCKS_AM if base_model.arch.get("AM", False) else LBLOCKS26)))
    )
    blocknum = (
        BLOCKIDZI if base_model.arch.get("ZI", False) else
        (BLOCKIDFLUX if base_model.arch.get("FLUX", False) else
         (BLOCKIDXLL if base_model.arch.get("XL", False) else
          (BLOCKIDAM if base_model.arch.get("AM", False) else BLOCKID)))
    )

    keymap = _build_keymap(base_model.theta)

    lam, p, scale = 1.5, 0.13, 0.2
    torch.manual_seed(0)

    lr_strs, lora_meta = [], {}

    for lora_model, ratio_str in lora_list:
        print(f"loading: {lora_model}")

        g, weights, deep = parse_ratio_spec(ratio_str, len(blocknum))
        lr_strs.append(ratio_str)

        lora_path = _resolve_lora_path(lora_model, model_path)
        lora_model_obj, meta, lisv2 = _load_lora(lora_path, device=device, cache_path=cache_path)
        lora_meta[lora_model_obj.sha256] = meta

        lw = merge_weights_inplace(
            lora_model_obj.theta,
            lisv2,
            base_model.arch.get("XL", False),
            blocks,
            p,
            lam,
            scale,
            [1.0],
        )
        
        plan = build_apply_plan(
            lw.keys(),
            arch=base_model.arch,
            mlv2=lisv2,
            keymap=keymap,
            theta_0=base_model.theta,
        )
        print(f"[dare-plan] {lora_model}: resolved={len(plan)}")

        for kind, down_k, up_k, alpha_k, tgt, part in tqdm(plan, desc=f"Applying {lora_model}...", leave=False):
            if down_k not in lw or up_k not in lw:
                continue

            target_key = tgt
            target_part = part
            
            if kind == "fused":
                W = base_model.theta.get(target_key)
                if not isinstance(W, torch.Tensor):
                    continue

                out = apply_zimage_lora_to_weight(
                    W,
                    part=target_part,
                    up=up,
                    down=down,
                    alpha=alpha,
                    ratio=ratio,
                )
                if out is not None:
                    base_model.theta[target_key] = out
                continue

            if kind == "zi":
                if (target_key is None) or (target_key not in base_model.theta):
                    target_key, target_part = zimage_resolve_target_any(down_k, zi_index)
                if target_key is None or target_key not in base_model.theta:
                    continue

            elif base_model.arch.get("AM", False):
                if (target_key is None) or (target_key not in base_model.theta):
                    target_key = anima_resolve_target_any(down_k, base_model.theta)
                if target_key is None or target_key not in base_model.theta:
                    continue

            else:
                if target_key is None or target_key not in base_model.theta:
                    continue

            ratio = _effective_ratio_for_target(
                target_key,
                g=g,
                weights=weights,
                deep=deep,
                blockids=blocknum,
                arch=base_model.arch,
            )

            down = lw[down_k]
            up = lw[up_k]
            dim = int(down.size(0))

            alpha = lw.get(alpha_k, dim)
            if alpha is None and isinstance(alpha_k, str) and alpha_k.endswith(".weight"):
                alpha = lw.get(alpha_k[:-len(".weight")], dim)
            if isinstance(alpha, torch.Tensor):
                alpha = float(alpha.item())
            alpha = float(dim if alpha is None else alpha)

            sc = float(alpha) / float(dim)

            if kind == "zi":
                W = base_model.theta.get(target_key)
                if not isinstance(W, torch.Tensor):
                    continue

                out = apply_zimage_lora_to_weight(
                    W,
                    part=target_part,
                    up=up,
                    down=down,
                    alpha=alpha,
                    ratio=ratio,
                )
                if out is not None:
                    base_model.theta[target_key] = out
                continue

            W = base_model.theta.get(target_key)
            if not isinstance(W, torch.Tensor):
                continue

            dev = W.device
            W32 = W.float()
            out = apply_lora_to_weight_inplace(
                W32,
                up.to(dev).float(),
                down.to(dev).float(),
                sc,
                ratio=ratio,
            )
            base_model.theta[target_key] = out.to(W.dtype)

        del lw, lora_model_obj.theta

    out_name = os.path.splitext(os.path.basename(output))[0]
    meta_new = {
        "sd_merge_models": json.dumps({
            "type": "pluslora-chattiori",
            "checkpoint_hash": base_model.sha256,
            "lora_hash": ",".join([k for k in lora_meta.keys() if k]),
            "alpha_info": "DARE:" + ",".join(lr_strs),
            "output_name": out_name,
        }),
        "checkpoint": json.dumps(base_model.metadata),
        "lora": json.dumps(lora_meta),
    }
    if args.memo is not None:
        meta_new["memo"] = args.memo

    baked = _make_checkpoint_model(base_model.theta, base_model, output=output, metadata=meta_new)
    print(f"Saving as {output}...")
    _save_umodel(baked, output, args=args)
    print(f"Done! ({round(os.path.getsize(output)/1073741824, 2)}G)")
    
def _infer_arch_from_lora_keys(keys: list[str]) -> tuple[bool, bool, bool]:
    """
    return (isxl, isflux, iszi)  heuristic
    """
    arch = {"XL": False, "FLUX": False, "ZI": False, "AM": False}

    # AM (Anima)
    for k in keys:
        if (
            "lora_unet_blocks_" in k
            or "lora_te_layers_" in k
            or re.match(r"^(?:model\.)?diffusion_model\.blocks\.\d+\..*\.lora_(?:A|B|down|up)\.weight$", k)
            or re.match(r"^(?:model\.)?diffusion_model\.final_layer\..*\.lora_(?:A|B|down|up)\.weight$", k)
            or re.match(r"^(?:model\.)?diffusion_model\.llm_adapter\.blocks\.\d+\..*\.lora_(?:A|B|down|up)\.weight$", k)
        ):
            arch["AM"] = True
            return arch
        
    # Z-Image / DiT
    for k in keys:
        if _ZI_LAYER_RE.match(k) or "diffusion_model.layers." in k or "diffusion_model_layers_" in k:
            arch["ZI"] = True
            return arch
        if "context_refiner" in k or "noise_refiner" in k:
            arch["ZI"] = True
            return arch

    # SDXL 
    for k in keys:
        if k.startswith("lora_te2_") or "te2_text_model" in k or "lora_te2_text_model_encoder_layers_" in k:
            arch["XL"] = True
            return arch

    # Flux
    for k in keys:
        if "text_encoders" in k or "qwen" in k:
            arch["FLUX"] = True
            return arch

    return arch


def _canonical_lora_base(down_k: str) -> str | None:
    """
    '...lora_A.weight' or '...lora_down.weight' -> base (without suffix)
    """
    if down_k.endswith(".lora_A.weight"):
        return down_k[:-len(".lora_A.weight")]
    if down_k.endswith(".lora_down.weight"):
        return down_k[:-len(".lora_down.weight")]
    return None


def _to_2d_up_down(up: torch.Tensor, down: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool, tuple]:
    """
    Returns (up2d [out,r], down2d [r,in_flat], is_conv, shape_info)
      shape_info:
        if conv: (in_ch, kh, kw)
        else:    (in_ch, 1, 1)
    """
    # up: (out, r) or (out, r, 1, 1)
    if up.ndim == 4:
        up2d = up.squeeze(3).squeeze(2)
        is_up_conv = True
    else:
        up2d = up
        is_up_conv = False

    # down: (r, in) or (r, in, kh, kw)
    if down.ndim == 4:
        r, in_ch, kh, kw = down.shape
        down2d = down.reshape(r, in_ch * kh * kw)
        is_down_conv = True
        shape_info = (in_ch, kh, kw)
    else:
        r, in_ch = down.shape
        down2d = down
        is_down_conv = False
        shape_info = (in_ch, 1, 1)

    is_conv = bool(is_up_conv or is_down_conv)
    return up2d, down2d, is_conv, shape_info


CLAMP_QUANTILE_DEFAULT = 0.99

@torch.inference_mode()
def _compress_uv_rankcap_dimstyle(U: torch.Tensor, V: torch.Tensor, r_cap: int) -> tuple[torch.Tensor, torch.Tensor]:
    k = int(U.shape[1])
    if (r_cap <= 0) or (k <= r_cap):
        return U, V

    Uc = U.to(dtype=torch.float32)
    Vc = V.to(dtype=torch.float32)

    # U = Qu Ru,  V^T = Qv Rv  => M = Qu (Ru Rv^T) Qv^T
    Qu, Ru = torch.linalg.qr(Uc, mode="reduced")       # Qu: [out,k], Ru: [k,k]
    Qv, Rv = torch.linalg.qr(Vc.t(), mode="reduced")   # Qv: [in,k],  Rv: [k,k]

    Msmall = Ru @ Rv.t()                               # [k,k]
    P, S, Vh = torch.linalg.svd(Msmall, full_matrices=False)

    r = min(int(r_cap), int(S.numel()))
    if r <= 0:
        return U[:, :0], V[:0, :]

    # up = (Qu P) diag(S)
    up = (Qu @ P[:, :r]) * S[:r]                       # [out,r]

    # down = (Vh Qv^T)
    down = (Vh[:r, :] @ Qv.t())                        # [r,in]

    return up, down


@torch.inference_mode()
def _clamp_uv_quantile_inplace(up: torch.Tensor, down: torch.Tensor, q: float) -> tuple[torch.Tensor, torch.Tensor]:
    if q is None:
        return up, down
    q = float(q)
    if not (0.0 < q < 1.0):
        return up, down

    dist = torch.cat([up.reshape(-1), down.reshape(-1)]).to(dtype=torch.float32)
    if dist.numel() == 0:
        return up, down

    hi = torch.quantile(dist.abs(), q)
    hi = float(hi.item()) if isinstance(hi, torch.Tensor) else float(hi)
    if not (hi > 0.0):
        return up, down

    lo = -hi
    up = up.clamp(lo, hi)
    down = down.clamp(lo, hi)
    return up, down


_SDXL_TE1_DIRECT_PREFIXES = (
    "text_encoders.clip_l.transformer.text_model.",
    "conditioner.embedders.0.model.transformer.text_model.",
)
_SDXL_TE2_DIRECT_PREFIXES = (
    "text_encoders.clip_g.transformer.text_model.",
    "conditioner.embedders.1.model.transformer.text_model.",
)

def _strip_any_model_prefix(s: str, prefixes: tuple[str, ...]) -> str | None:
    for p in prefixes:
        if s.startswith(p):
            return s[len(p):]
    return None

def _direct_unet_base_to_kohya(base: str) -> str | None:
    b = base
    if b.startswith("model.diffusion_model."):
        b = b[len("model.diffusion_model."):]
    elif b.startswith("diffusion_model."):
        b = b[len("diffusion_model."):]
    else:
        return None
    return "lora_unet_" + b.replace(".", "_")

def _direct_sdxl_textenc_base_to_kohya(base: str) -> str | None:
    rest = _strip_any_model_prefix(base, _SDXL_TE1_DIRECT_PREFIXES)
    if rest is not None:
        if rest.startswith("encoder.layers."):
            m = re.match(r"^encoder\.layers\.(\d+)\.(.+)$", rest)
            if m:
                return f"lora_te1_text_model_encoder_layers_{m.group(1)}_{m.group(2).replace('.', '_')}"
        if rest.startswith("embeddings."):
            return "lora_te1_text_model_" + rest.replace(".", "_")
        if rest == "final_layer_norm":
            return "lora_te1_text_model_final_layer_norm"

    rest = _strip_any_model_prefix(base, _SDXL_TE2_DIRECT_PREFIXES)
    if rest is not None:
        if rest.startswith("encoder.layers."):
            m = re.match(r"^encoder\.layers\.(\d+)\.(.+)$", rest)
            if m:
                return f"lora_te2_text_model_encoder_layers_{m.group(1)}_{m.group(2).replace('.', '_')}"
        if rest.startswith("embeddings."):
            return "lora_te2_text_model_" + rest.replace(".", "_")
        if rest == "final_layer_norm":
            return "lora_te2_text_model_final_layer_norm"

    if base in ("text_encoders.clip_l.text_projection", "conditioner.embedders.0.model.text_projection"):
        return "lora_te1_text_projection"
    if base in ("text_encoders.clip_g.text_projection", "conditioner.embedders.1.model.text_projection"):
        return "lora_te2_text_projection"

    return None

def _normalized_compvis_to_kohya_base(full: str) -> str | None:
    if full.startswith("diffusion_model_input_blocks_"):
        return "lora_unet_" + full[len("diffusion_model_"):]
    if full.startswith("diffusion_model_middle_block_"):
        return "lora_unet_" + full[len("diffusion_model_"):]
    if full.startswith("diffusion_model_output_blocks_"):
        return "lora_unet_" + full[len("diffusion_model_"):]
    if full.startswith("diffusion_model_out_"):
        return "lora_unet_" + full[len("diffusion_model_"):]

    if full.startswith("0_transformer_text_model_"):
        return "lora_te1_" + full[len("0_transformer_"):]
    if full.startswith("transformer_text_model_"):
        return "lora_te1_" + full[len("transformer_"):]

    return None

def canonical_kohya_base_from_down_key(down_k: str, *, arch: dict, lisv2: bool) -> str | None:
    base = _canonical_lora_base(down_k)
    if base is None:
        return None

    # already kohya-style
    if base.startswith(("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_", "lora_vae_", "lora_first_stage_model_")):
        return base

    # direct unet
    k = _direct_unet_base_to_kohya(base)
    if k is not None:
        return k

    # direct sdxl text encoder
    if arch.get("XL", False):
        k = _direct_sdxl_textenc_base_to_kohya(base)
        if k is not None:
            return k

    # fallback through normalized converted name
    full = convert_diffusers_name_to_compvis_cached(down_k, bool(lisv2))
    k = _normalized_compvis_to_kohya_base(full)
    if k is not None:
        return k

    # last resort: keep original stripped base
    return base


@torch.inference_mode()
def merge_loras_only(
    lora_list: list[list[str]],
    output: str,
    model_path: str,
    device: str = "cpu",
    *,
    merge_rank: int = 64,
    arch_set: str = "auto",
    merge_norm="none",
    merge_scale=1.0,
    unet_only=False,
    clamp_quantile: float = CLAMP_QUANTILE_DEFAULT,
    intermediate_mult: int = 4,
):
    cache_path = os.path.join(base_path(), "cache.json")

    model_path = normalize_path(model_path)
    if not lora_list:
        return "ERROR: No LoRA Selected"

    print("Merge LoRAs start")
    print(f"Output: {output}")
    print(f"merge_rank(target): {merge_rank}  (0=keep as-is per module)")
    print(f"arch: {arch_set}")
    print(f"merge_norm: {merge_norm}  merge_scale: {merge_scale}")
    print(f"clamp_quantile: {clamp_quantile}  intermediate_mult: {intermediate_mult}")

    # arch decision
    arch = {"XL": False, "FLUX": False, "ZI": False, "AM": False}
    decided = False
    blocknum = None

    merged_meta = {}
    merged_sources = []

    n_src = max(1, len(lora_list))
    if merge_norm == "mean":
        global_norm = 1.0 / n_src
    elif merge_norm == "sqrt":
        global_norm = 1.0 / math.sqrt(n_src)
    else:
        global_norm = 1.0
    global_norm *= float(merge_scale)

    # base -> state(U,V,is_conv,shape_info,out_ch,in_flat)
    merged: dict[str, dict] = {}

    # intermediate keep rank
    k_soft = 0
    if merge_rank > 0:
        k_soft = max(int(merge_rank), 1) * max(int(intermediate_mult), 1)

    for lora_model, ratio_str in lora_list:
        lpath = _resolve_lora_path(lora_model, model_path)
        print(f"loading lora: {lora_model}")

        lora_obj, meta, lisv2 = _load_lora(lpath, device=device, cache_path=cache_path)
        merged_meta[lora_obj.sha256] = meta
        merged_sources.append({"name": lora_model, "hash": lora_obj.sha256, "ratio": ratio_str})

        if not decided:
            if arch_set == "auto":
                arch = _infer_arch_from_lora_keys(list(lora_obj.theta.keys()))
            else:
                arch["XL"] = (arch_set == "sdxl")
                arch["FLUX"] = (arch_set == "flux")
                arch["ZI"] = (arch_set == "zi")
                arch["AM"] = (arch_set == "am")

            blocknum = (
                BLOCKIDZI if arch["ZI"] else
                (BLOCKIDFLUX if arch["FLUX"] else
                (BLOCKIDXLL if arch["XL"] else
                (BLOCKIDAM if arch["AM"] else BLOCKID)))
            )
            decided = True
            print(f"detected arch: isxl={arch['XL']} isflux={arch['FLUX']} iszi={arch['ZI']} isam={arch['AM']} (blocklen={len(blocknum)})")

        g, weights, deep = parse_ratio_spec(ratio_str, len(blocknum))

        for down_k in tqdm(list(_iter_lora_down_keys(lora_obj.theta)), desc=f"Merging factors {lora_model}..."):
            d, u, a = _pair_from_down_key(down_k)
            if d is None:
                continue
            if (u not in lora_obj.theta) or (d not in lora_obj.theta):
                continue

            base = canonical_kohya_base_from_down_key(
                d,
                arch=arch,
                lisv2=bool(lisv2),
            )
            if base is None:
                continue

            if unet_only:
                if ("lora_te" in base) or ("text_encoder" in base) or ("te_" in base):
                    continue

            down = lora_obj.theta[d]
            up = lora_obj.theta[u]
            if not (isinstance(down, torch.Tensor) and isinstance(up, torch.Tensor)):
                continue

            rank = int(down.size(0))
            alpha = lora_obj.theta.get(a, rank)
            if alpha is None and isinstance(a, str) and a.endswith(".weight"):
                alpha = lora_obj.theta.get(a[:-len(".weight")], rank)
            if isinstance(alpha, torch.Tensor):
                alpha = float(alpha.item())
            alpha = float(rank if alpha is None else alpha)

            # ratio target key (block/elementals)
            if arch["ZI"]:
                tgt, _part = zimage_resolve_target(d)
                target_key = tgt if tgt is not None else d
            elif arch["AM"]:
                target_key = _am_target_key_from_lora_down(d) or d
            else:
                full = convert_diffusers_name_to_compvis_cached(d, bool(lisv2))
                msd = full.split(".", 1)[0]
                if arch["XL"]:
                    msd = msd.replace("lora_unet", "diffusion_model").replace("lora_te1_text_model", "0_transformer_text_model")
                target_key = msd

            ratio_eff = _effective_ratio_for_target(
                target_key,
                g=g, weights=weights, deep=deep,
                blockids=blocknum,
                arch=arch,
            )

            # bake scale = ratio * (alpha/rank)
            s = global_norm * float(ratio_eff) * (float(alpha) / float(rank))
            if s == 0.0:
                continue

            up2d, down2d, is_conv, shape_info = _to_2d_up_down(up, down)
            if up2d.ndim != 2 or down2d.ndim != 2:
                continue
            out_ch = int(up2d.shape[0])
            r_up = int(up2d.shape[1])
            r_dn = int(down2d.shape[0])
            if r_up != r_dn:
                continue
            in_flat = int(down2d.shape[1])

            sign = -1.0 if s < 0 else 1.0
            fac = math.sqrt(abs(float(s)))

            U_add = up2d.to(dtype=torch.float32, device="cpu") * fac
            V_add = down2d.to(dtype=torch.float32, device="cpu") * (fac * sign)

            st = merged.get(base)
            if st is None:
                merged[base] = {
                    "U": U_add,
                    "V": V_add,
                    "is_conv": bool(is_conv),
                    "shape_info": shape_info,
                    "out_ch": out_ch,
                    "in_flat": in_flat,
                }
            else:
                if int(st["out_ch"]) != out_ch or int(st["in_flat"]) != in_flat:
                    continue

                st["U"] = torch.cat([st["U"], U_add], dim=1)  # out x (k+rank)
                st["V"] = torch.cat([st["V"], V_add], dim=0)  # (k+rank) x in

                if (merge_rank > 0) and (k_soft > 0) and (int(st["U"].shape[1]) > k_soft):
                    st["U"], st["V"] = _compress_uv_rankcap_dimstyle(st["U"], st["V"], int(k_soft))

        del lora_obj

    # finalize + save
    out_sd = {}

    for base, st in tqdm(list(merged.items()), desc="Finalizing merged LoRA..."):
        U = st["U"]
        V = st["V"]
        is_conv = bool(st["is_conv"])
        out_ch = int(st["out_ch"])
        in_flat = int(st["in_flat"])

        k = int(U.shape[1])
        if k <= 0:
            continue
        if merge_rank > 0:
            U, V = _compress_uv_rankcap_dimstyle(U, V, int(merge_rank))
            r = int(U.shape[1])
            if r < int(merge_rank):
                pad = int(merge_rank) - r
                U = torch.cat([U, torch.zeros((U.shape[0], pad), dtype=U.dtype, device=U.device)], dim=1)
                V = torch.cat([V, torch.zeros((pad, V.shape[1]), dtype=V.dtype, device=V.device)], dim=0)
                r = int(merge_rank)
        else:
            r = int(U.shape[1])

        # clamp
        U, V = _clamp_uv_quantile_inplace(U, V, clamp_quantile)

        # reshape back to kohya style
        if is_conv:
            in_ch, kh, kw = st["shape_info"]
            down_t = V.reshape(r, int(in_ch), int(kh), int(kw))
            up_t = U.reshape(out_ch, r, 1, 1)
        else:
            down_t = V.reshape(r, in_flat)
            up_t = U.reshape(out_ch, r)

        # save dtype
        up_t = up_t.contiguous().cpu().to(torch.float16)
        down_t = down_t.contiguous().cpu().to(torch.float16)

        # output keys (kohya)
        down_k = base + ".lora_down.weight"
        up_k = base + ".lora_up.weight"
        alpha_k = base + ".alpha"

        out_sd[down_k] = down_t
        out_sd[up_k] = up_t
        out_sd[alpha_k] = torch.tensor(float(r), dtype=torch.float32)

    meta_new = {
        "sd_merge_models": json.dumps({
            "type": "lora-merge-chattiori-dim",
            "merge_rank_target": int(merge_rank),
            "intermediate_mult": int(intermediate_mult),
            "clamp_quantile": float(clamp_quantile),
            "arch": ("zi" if arch["ZI"] else ("flux" if arch["FLUX"] else ("sdxl" if arch["XL"] else ("am" if arch["AM"] else "sd")))),
            "merge_norm": str(merge_norm),
            "merge_scale": float(merge_scale),
            "sources": merged_sources,
            "output_name": os.path.splitext(os.path.basename(output))[0],
        }),
        "lora": json.dumps(merged_meta),
    }
    
    merged_lora_model = _make_lora_model(out_sd, output=output, metadata=meta_new, arch=arch)
    print(f"Saving merged LoRA as {output}...")
    _save_umodel(merged_lora_model, output, args=args)

    print(f"Done! {round(os.path.getsize(output)/1073741824, 3)}G")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge several loras to checkpoint")
    parser.add_argument("model_path", type=str, help="Path to models")
    parser.add_argument("checkpoint", type=str, nargs="?", help="Name of the checkpoint", default=None)
    parser.add_argument("loras", type=str, nargs="?", help="Path and alpha of LoRAs eg.)\"Path:alpha,Path:alpha, ...\"", default=None)
    parser.add_argument("--save_half", action="store_true", help="Save as float16", required=False)
    parser.add_argument("--save_bhalf", action="store_true", help="Save as bfloat16", required=False)
    parser.add_argument("--prune", action="store_true", help="Prune Model", required=False)
    parser.add_argument("--save_quarter", action="store_true", help="Save as float8", required=False)
    parser.add_argument("--keep_ema", action="store_true", help="Keep ema", required=False)
    parser.add_argument("--dare", action="store_true", help="Use DARE Merge")
    parser.add_argument("--merge_loras", action="store_true",
                        help="Merge multiple LoRAs into a single LoRA (does NOT bake into checkpoint)")
    parser.add_argument("--merge_rank", type=int, default=64,
                        help="Rank cap for merged LoRA. 0 means unlimited (exact concat; can get huge). Default=64")
    parser.add_argument("--merge_arch", type=str, default="auto",
                        choices=["auto", "sd", "sdxl", "flux", "zi", "am"],
                        help="Architecture for ratio/block mapping in merge mode. Default=auto")
    parser.add_argument("--merge_norm", type=str, default="none",
                    choices=["none","sqrt","mean"],
                    help="Global normalization for merging many LoRAs. mean=1/N, sqrt=1/sqrt(N).")
    parser.add_argument("--merge_scale", type=float, default=1.0,
                        help="Extra global scale multiplier applied after merge_norm.")
    parser.add_argument("--merge_unet_only", action="store_true",
                        help="Only merge UNet LoRA modules (skip text encoders).")
    parser.add_argument("--merge_clamp_q", type=float, default=CLAMP_QUANTILE_DEFAULT,
                        help="Quantile clamp applied to merged LoRA factors before save. Default=0.99")
    parser.add_argument("--merge_intermediate_mult", type=int, default=4,
                        help="Temporary rank multiplier before final compression. Default=4")
    parser.add_argument("--bake_clip_scale", type=float, default=1.0,
                    help="Global scale multiplier for text encoder LoRA modules when baking into checkpoint. Default=1.0 (no extra scaling).")
    parser.add_argument("--bake_unet_only", action="store_true",
                    help="Bake only UNet/DiT modules (skip text encoder LoRA modules).")
    parser.add_argument("--bake_norm", type=str, default="sqrt", choices=["none","sqrt","mean"],
                        help="Global normalization when baking many LoRAs. sqrt=1/sqrt(N), mean=1/N.")
    parser.add_argument("--bake_scale", type=float, default=1.0,
                        help="Extra global multiplier applied after bake_norm.")
    parser.add_argument("--bake_rank_cap", type=int, default=0,
                        help="Per-module rank cap before applying. 0 disables.")
    parser.add_argument("--bake_clamp_q", type=float, default=0.0,
                        help="Quantile clamp for up/down weights (0 disables). Suggested 0.995~0.999.")
    parser.add_argument("--bake_delta_cap", type=float, default=0.0,
                        help="Per-module cap: ||ΔW||_F <= cap * ||W||_F (0 disables). Suggested 0.02~0.10.")
    parser.add_argument("--bake_fp32", action="store_true",
                        help="Accumulate LoRA deltas in fp32 then cast back to original dtype.")
    parser.add_argument("--bake_guard", type=str, default="auto",
                        choices=["none", "auto", "cap"],
                        help="Safety guard for baking. auto: enable cap when baking many LoRAs and bake_delta_cap<=0. cap: force bake_guard_cap. none: disable.")
    parser.add_argument("--bake_guard_cap", type=float, default=0.05,
                        help="Cap for relative update: ||ΔW||_F <= cap * ||W||_F (used by bake_guard). Suggested 0.02~0.10.")
    parser.add_argument("--bake_guard_skip", type=float, default=0.25,
                        help="Skip module if estimated ||ΔW||_F / ||W||_F exceeds this. 0 disables. Suggested 0.15~0.50.")
    parser.add_argument("--bake_budget_report", action="store_true",
        help="Print budget scaling report (top targets by shrink).")
    parser.add_argument("--no_metadata", action="store_true", help="Save without metadata")
    parser.add_argument("--memo",   type=str,   help="Additional info bake in metadata", default=None)
    parser.add_argument("--save_safetensors", action="store_true", help="Save as .safetensors", required=False)
    parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
    parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
    args = parser.parse_args()
    args.model_path = normalize_path(args.model_path)

    ll  = get_loralist(args.loras, model_path=args.model_path)
    out = normalize_path(os.path.join(args.model_path, f"{args.output}.{'safetensors' if args.save_safetensors else 'ckpt'}"))
    
    # --- LoRA merge only mode (no checkpoint bake) ---
    if getattr(args, "merge_loras", False):
        out_lora = normalize_path(os.path.join(
            args.model_path,
            f"{args.output}.safetensors"
        ))
        merge_loras_only(
            ll,
            out_lora,
            args.model_path,
            args.device,
            merge_rank=int(getattr(args, "merge_rank", 64)),
            arch_set=str(getattr(args, "merge_arch", "auto")),
            merge_norm=str(getattr(args, "merge_norm","none")), 
            merge_scale=float(getattr(args, "merge_scale", 1.0)), 
            unet_only=getattr(args, "merge_unet_only", False),
            clamp_quantile=float(getattr(args, "merge_clamp_q", CLAMP_QUANTILE_DEFAULT)),
            intermediate_mult=int(getattr(args, "merge_intermediate_mult", 4)),
        )
        raise SystemExit(0)
    else:
        if args.checkpoint is None:
            raise ValueError("checkpoint is None")
        
    if args.dare:
        mainlora = _resolve_lora_path(ll[0][0], args.model_path)
        darelora(mainlora, ll, args.checkpoint, out, args.model_path, args.device)
    else:
        pluslora(ll, args.checkpoint, out, args.model_path, args.device)