import json
import torch
import safetensors.torch
import os
import filelock
import hashlib
import re

BLOCKID26=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID17=["BASE","IN01","IN02","IN04","IN05","IN07","IN08","M00","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID12=["BASE","IN04","IN05","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05"]
BLOCKID20=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08"]
BLOCKNUMS = [12,17,20,26]
BLOCKIDS=[BLOCKID12,BLOCKID17,BLOCKID20,BLOCKID26]
LBLOCKS26=["encoder",
"diffusion_model_input_blocks_0_",
"diffusion_model_input_blocks_1_",
"diffusion_model_input_blocks_2_",
"diffusion_model_input_blocks_3_",
"diffusion_model_input_blocks_4_",
"diffusion_model_input_blocks_5_",
"diffusion_model_input_blocks_6_",
"diffusion_model_input_blocks_7_",
"diffusion_model_input_blocks_8_",
"diffusion_model_input_blocks_9_",
"diffusion_model_input_blocks_10_",
"diffusion_model_input_blocks_11_",
"diffusion_model_middle_block_",
"diffusion_model_output_blocks_0_",
"diffusion_model_output_blocks_1_",
"diffusion_model_output_blocks_2_",
"diffusion_model_output_blocks_3_",
"diffusion_model_output_blocks_4_",
"diffusion_model_output_blocks_5_",
"diffusion_model_output_blocks_6_",
"diffusion_model_output_blocks_7_",
"diffusion_model_output_blocks_8_",
"diffusion_model_output_blocks_9_",
"diffusion_model_output_blocks_10_",
"diffusion_model_output_blocks_11_",
"embedders"]
checkpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}

checkpoint_dict_skip_on_merge = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]


re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
re_compiled = {}

suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "norm1": "in_layers_0",
        "norm2": "out_layers_0",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}

def convert_diffusers_name_to_compvis(key, is_sd2):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_conv_in(.*)"):
        return f'diffusion_model_input_blocks_0_0{m[0]}'

    if match(m, r"lora_unet_conv_out(.*)"):
        return f'diffusion_model_out_2{m[0]}'

    if match(m, r"lora_unet_time_embedding_linear_(\d+)(.*)"):
        return f"diffusion_model_time_embed_{m[0] * 2 - 2}{m[1]}"

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    if match(m, r"lora_te2_text_model_encoder_layers_(\d+)_(.+)"):
        if 'mlp_fc1' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
        elif 'mlp_fc2' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
        else:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

    return key

def transform_checkpoint_dict_key(k):
  for text, replacement in checkpoint_dict_replacements.items():
      if k.startswith(text):
          k = replacement + k[len(text):]

  return k
def get_state_dict_from_checkpoint(pl_sd):
  pl_sd = pl_sd.pop("state_dict", pl_sd)
  pl_sd.pop("state_dict", None)

  sd = {}
  for k, v in pl_sd.items():
      new_key = transform_checkpoint_dict_key(k)

      if new_key is not None:
          sd[new_key] = v

  pl_sd.clear()
  pl_sd.update(sd)

  return pl_sd

def read_metadata_from_safetensors(filename):
    import json

    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"
        json_data = json_start + file.read(metadata_len-2)
        json_obj = json.loads(json_data)

        res = {}
        for k, v in json_obj.get("__metadata__", {}).items():
            res[k] = v
            if isinstance(v, str) and v[0:1] == '{':
                try:
                    res[k] = json.loads(v)
                except Exception as e:
                    pass

        return res

def load_model(model_path, device="cpu"):
    if os.path.splitext(model_path)[1] == ".safetensors":
        model = safetensors.torch.load_file(model_path, device=device)
    else:
        model = torch.load(model_path, map_location=device)
    sd = get_state_dict_from_checkpoint(model)
    metadata = read_metadata_from_safetensors(model_path)
    return sd, metadata

def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    """
    This method locks the file. see https://github.com/huggingface/safetensors/issues/164
    If the file isn't .safetensors or doesn't have metadata, return empty dict.
    """
    if os.path.splitext(safetensors_file)[1] != ".safetensors":
        return {}

    with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return metadata

def load_state_dict(file_name, dtype, device = "cpu"):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = safetensors.torch.load_file(file_name,device=device)
        metadata = load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location=device)
        metadata = {}

    isv2 = False

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype = dtype, device = device)
            if "resblocks" in key:
                isv2 = True

    if isv2: print("SD2.X")

    return sd, metadata, isv2
cache_data = None
def pluslora(lora_list: list,model,output,device="cpu"):
    cache_filename = os.path.join(model_path, "cache.json")
    def cache(subsection):
        global cache_data
        if cache_data is None:
            with filelock.FileLock(f"{cache_filename}.lock"):
                if not os.path.isfile(cache_filename):
                    cache_data = {}
                else:
                    with open(cache_filename, "r", encoding="utf8") as file:
                        cache_data = json.load(file)

        s = cache_data.get(subsection, {})
        cache_data[subsection] = s

        return s

    def dump_cache():
        with filelock.FileLock(f"{cache_filename}.lock"):
            with open(cache_filename, "w", encoding="utf8") as file:
                json.dump(cache_data, file, indent=4)

    def sha256(filename, title):
        hashes = cache("hashes")

        sha256_value = sha256_from_cache(filename, title, 1)
        if sha256_value is not None:
            return sha256_value

        print(f"Calculating sha256 for {filename}: ", end='')
        sha256_value = calculate_sha256(filename)
        print(f"{sha256_value}")

        hashes[title] = {
            "mtime": os.path.getmtime(filename),
            "sha256": sha256_value,
        }

        dump_cache()

        return sha256_value

    def calculate_shorthash(filename):
        sha256 = sha256(filename, f"checkpoint/{os.path.splitext(os.path.basename(filename))[0]}")
        if sha256 is None:
            return

        shorthash = sha256[0:10]

        return shorthash

    def calculate_sha256(filename):
        hash_sha256 = hashlib.sha256()
        blksize = 1024 * 1024

        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(blksize), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()


    def sha256_from_cache(filename, title, par = 0):
        hashes = cache("hashes")
        ondisk_mtime = os.path.getmtime(filename)

        if title not in hashes:
          if par == 0:
            sh = sha256(filename, title)
          else:
            return None

        cached_sha256 = hashes[title].get("sha256", None)
        cached_mtime = hashes[title].get("mtime", 0)

        if ondisk_mtime > cached_mtime or cached_sha256 is None:
            return None

        return cached_sha256
    if model == []: return "ERROR: No model Selected"
    if lora_model == []: return "ERROR: No LoRA Selected"

    add = ""
    print("Plus LoRA start")
    import lora

    print(f"Loading {model}")
    theta_0, metadata = load_model(model, device=device)
    model_name = os.path.splitext(os.path.basename(model))[0]
    isxl = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in theta_0.keys()
    isv2 = "cond_stage_model.model.transformer.resblocks.0.attn.out_proj.weight" in theta_0.keys()

    keychanger = {}
    for key in theta_0.keys():
        if "model" in key:
            skey = key.replace(".","_").replace("_weight","")
            if "conditioner_embedders_" in skey:
                keychanger[skey.split("conditioner_embedders_",1)[1]] = key
            else:
                if "wrapped" in skey:
                    keychanger[skey.split("wrapped_",1)[1]] = key
                else:
                    keychanger[skey.split("model_",1)[1]] = key
    lr=[]
    lh={}
    for lora_model, loraratio in lora_list:
        print(f"loading: {lora_model}")
        loraratios=[float(x) for x in loraratio.replace(" ","").split(",")]
        lr.append("["+",".join(loraratios)+"]")
        
        lora_sd, lora_metadata, lisv2 = load_state_dict(lora_model, torch.float)
        lora_name = os.path.splitext(os.path.basename(lora_model))[0]
        lora_hash = sha256_from_cache(lora_model, f"lora/{lora_name}")
        lh[lora_hash]=lora_metadata

        print(f"merging..." ,lora_model)
        for key in lora_sd.keys():
            
            ratio = loraratios[0]

            import lora
            fullkey = convert_diffusers_name_to_compvis(key,lisv2)
            #print(fullkey)
            msd_key = fullkey.split(".", 1)[0]
            if isxl:
                if "lora_unet" in msd_key:
                    msd_key = msd_key.replace("lora_unet", "diffusion_model")
                elif "lora_te1_text_model" in msd_key:
                    msd_key = msd_key.replace("lora_te1_text_model", "0_transformer_text_model")

            for i,block in enumerate(LBLOCKS26):
                if block in fullkey or block in msd_key:
                    try:
                        ratio = loraratios[i]
                    except:
                        ratio = loraratios[0]
            if msd_key not in keychanger.keys():
                  continue
            if "lora_down" in key:
                up_key = key.replace("lora_down", "lora_up")
                alpha_key = key[:key.index("lora_down")] + 'alpha'

                # print(f"apply {key} to {module}")

                down_weight = lora_sd[key].to(device="cpu")
                up_weight = lora_sd[up_key].to(device="cpu")

                dim = down_weight.size()[0]
                alpha = lora_sd.get(alpha_key, dim)
                scale = alpha / dim
                # W <- W + U * D
                
                weight = theta_0[keychanger[msd_key]].to(device="cpu")

                if len(weight.size()) == 2:
                    # linear
                    weight = weight + ratio * (up_weight @ down_weight) * scale

                elif down_weight.size()[2:4] == (1, 1):
                    # conv2d 1x1
                    weight = (
                        weight
                        + ratio
                        * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                        * scale
                    )
                else:
                    # conv2d 3x3
                    conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                    # print(conved.size(), weight.size(), module.stride, module.padding)
                    weight = weight + ratio * conved * scale
                theta_0[keychanger[msd_key]] = torch.nn.Parameter(weight)
    #usemodelgen(theta_0,model)
    output_name = os.path.splitext(os.path.basename(output))[0]
    new_metadata = {"sd_merge_models": {}, "checkpoint": {}, "lora": {}}
    merge_recipe = {
        "type": "pluslora-chattiori", # indicate this model was merged with chattiori's model mereger
        "checkpoint_hash": sha256_from_cache(model, f"checkpoint/{model_name}"),
        "lora_hash": ",".join(lh.keys()),
        "alpha_info": ",".join(lr),
        "output_name": output_name,
        }
    new_metadata["sd_merge_models"] = json.dumps(merge_recipe)
    new_metadata["checkpoint"] = json.dumps(metadata)
    for hs, mt in lh.items():
        new_metadata["lora"][hs] = mt
    new_metadata["lora"] = json.dumps(new_metadata["lora"])
    safetensors.torch.save_file(theta_0, output, metadata=new_metadata)

    del theta_0
    return "Done"
