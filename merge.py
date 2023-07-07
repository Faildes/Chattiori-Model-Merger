import glob
import os
import numpy as np
import sys
import time
import hashlib
import json
import datetime
import csv

import filelock
import copy
import argparse
import torch
import torch.nn as nn
import scipy.ndimage
from scipy.ndimage import median_filter as filter
import re
import shutil
import safetensors.torch
import safetensors
import random
from tqdm import tqdm

FINETUNEX = ["IN","OUT","OUT2","CONT","COL1","COL2","COL3"]
NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS
blockid=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]

def tagdict(presets):
    presets=presets.splitlines()
    wdict={}
    for l in presets:
        w=[]
        if ":" in l :
            key = l.split(":",1)[0]
            w = l.split(":",1)[1]
        if "\t" in l:
            key = l.split("\t",1)[0]
            w = l.split("\t",1)[1]
        if len([w for w in w.split(",")]) == 26:
            wdict[key.strip()]=w
    return wdict
path_root = os.getcwd()
userfilepath = os.path.join(path_root, "mbwpresets.txt")
if os.path.isfile(userfilepath):
    try:
        with open(userfilepath) as f:
            weights_presets = f.read()
            filepath = userfilepath
    except OSError as e:
            pass
else:
    filepath = os.path.join(path_root, "mbwpresets_master.txt")
    try:
        with open(filepath) as f:
            weights_presets = f.read()
            shutil.copyfile(filepath, userfilepath)
    except OSError as e:
            pass
weights_presets_list = tagdict(weights_presets)

deep_a = []
deep_b = []
useblocks = False
def wgta(string):
    global deep_a
    if type(string) == int or type(string) == float:
        return float(string)
    elif type(string) == list:
        useblocks = True
        string, deep_a = deepblock(string)
        while type(string) == list and len(string) == 1:
          string = string[0]
        return string
    elif type(string) == str:
        useblocks = True
        string, deep_a = deepblock([string])
        while type(string) == list and len(string) == 1:
          string = string[0]
        return string

def wgtb(string):
    global deep_b
    if type(string) == int or type(string) == float:
        return float(string)
    elif type(string) == list:
        useblocks = True
        string, deep_b = deepblock(string)
        while type(string) == list and len(string) == 1:
          string = string[0]
        return string
    elif type(string) == str:
        useblocks = True
        string, deep_b = deepblock([string])
        while type(string) == list and len(string) == 1:
          string = string[0]
        return string

def deepblock(string):
    res1 = []
    res2 = []
    for x in string:
        try:
            get = weights_presets_list[x].replace("\n", ",").split(",")
            for a in get:
                res1.append(float(a))
        except KeyError:
            try:
                x = float(x)
                res1.append(x)
            except:
                if type(x) is not str: continue
                bard = x.replace("\n", ",").split(",")
                if len(bard) == 0: continue
                elif len(bard) == 1: res2.append(bard[0])
                else:
                    cor1, cor2 = deepblock(bard)
                    res1.extend(cor1)
                    res2.extend(cor2)
    return res1, res2

def rand_ratio(string):
    if type(string) is not str:
        print(f"ERROR: illegal rand ratio: {string}") 
        exit()
    deep_res = []
    if "[" in string:
        tram = string.split("[")
        string = tram[0]
        deep = tram[1].replace("]","")
    else:
      deep = []
    parsed = [a for a in string.replace("\n",",").replace(","," ").split(" ") if (a != "" and a != " ")]
    if type(parsed) is list:
        try:
            rmin = float(parsed[0])
        except ValueError: rmin = 0.0
        try:
            try:
                rmax = float(parsed[1])
            except ValueError: rmax = 1.0
        except IndexError: rmax = 1.0
        try:
            try:
                seed = int(parsed[2])
            except ValueError: seed = random.randint(1, 4294967295)
        except IndexError: seed = random.randint(1, 4294967295)
    else:
        rmin = 0.0
        rmax = 1.0
        seed = random.randint(1, 4294967295)
    np.random.seed(seed)
    ratios = np.random.uniform(rmin, rmax, (1, 26))
    ratios = ratios[0].tolist()
    if len(deep) > 0:
        deep = deep.replace("\n",",")
        deep = deep.split(",")
        for d in deep:
            if "PRESET" in d:
                res1 = []
                preset_pack = d.split(":")[1]
                preset_name, drat = float(preset_pack.split("(")[0]), preset_pack.split("(")[1].replace(")","")
                try:
                    get = weights_presets_list[preset_name].replace("\n", ",").split(",")
                    for a in get:
                        res1.append(float(a))
                    dr1 = ratios
                    for d in range(0, 25):
                        ratios[d] = dr1[d] * (1 - drat) + res1[d] * drat
                    continue
                except KeyError: continue
            if d.count(":") != 2 :continue
            dbs,dws,dr = d.split(":")[0],d.split(":")[1],d.split(":")[2]
            dbs = dbs.split(" ")
            if dws == "ALL":
                if "(" in dr:
                    dr0, drat = float(dr.split("(")[0]), dr.split("(")[1].replace(")","")
                    for db in dbs:
                        dr1 = ratios[blockid.index(db)]
                        dr = dr1 * (1 - drat) + dr0 * drat
                        ratios[blockid.index(db)] = dr
                else:
                    for db in dbs:
                        ratios[blockid.index(db)] = float(dr)
            else:
                if "(" in dr:
                    dr0, drat = float(dr.split("(")[0]), dr.split("(")[1].replace(")","")
                    for db in dbs:
                        dr1 = ratios[blockid.index(db)]
                        dr = dr1 * (1 - drat) + dr0 * drat
                        deep_res.append(f"{db}:{dws}:{dr}")
                else:
                    dr0 = float(dr)
                    for db in dbs:
                        deep_res.append(f"{db}:{dws}:{dr0}")
    return ratios, seed, deep_res

def fineman(fine):
    fine = [
        1 - fine[0] * 0.01,
        1+ fine[0] * 0.02,
        1 - fine[1] * 0.01,
        1+ fine[1] * 0.02,
        1 - fine[2] * 0.01,
        [x*0.02 for x in fine[3:]]
                ]
    return fine

FINETUNES = [
"model.diffusion_model.input_blocks.0.0.weight",
"model.diffusion_model.input_blocks.0.0.bias",
"model.diffusion_model.out.0.weight",
"model.diffusion_model.out.0.bias",
"model.diffusion_model.out.2.weight",
"model.diffusion_model.out.2.bias",
]

parser = argparse.ArgumentParser(description="Merge two or three models")
parser.add_argument("mode", choices=["WS","AD","NoIn","TRS","ST","sAD","TD","TS","SIG","GEO","MAX","RM"], help="Merging mode")
parser.add_argument("model_path", type=str, help="Path to models")
parser.add_argument("model_0", type=str, help="Name of model 0")
parser.add_argument("model_1", type=str, help="Optional, Name of model 1", default=None)
parser.add_argument("--model_2", type=str, help="Optional, Name of model 2", default=None, required=False)
parser.add_argument("--m0_name", type=str, help="Custom name of model 0", default=None, required=False)
parser.add_argument("--m1_name", type=str, help="Custom name of model 1", default=None, required=False)
parser.add_argument("--m2_name", type=str, help="Custom name of model 2", default=None, required=False)
parser.add_argument("--vae", type=str, help="Path to vae", default=None, required=False)
parser.add_argument("--use_dif_10", action="store_true", help="Use the difference of model 1 and model 0 as model 1", required=False)
parser.add_argument("--use_dif_20", action="store_true", help="Use the difference of model 2 and model 0 as model 2", required=False)
parser.add_argument("--use_dif_21", action="store_true", help="Use the difference of model 2 and model 1 as model 2", required=False)
parser.add_argument("--alpha", type=wgta, help="Alpha value, optional, defaults to 0", default=0.0, required=False)
parser.add_argument("--rand_alpha", type=str, help="Random Alpha value, optional", default=None, required=False)
parser.add_argument("--beta", type=wgtb, help="Beta value, optional, defaults to 0", default=0.0, required=False)
parser.add_argument("--rand_beta", type=str, help="Random Beta value, optional", default=None, required=False)
parser.add_argument("--cosine0", action="store_true", help="Favors model 0's structure with details from 1", required=False)
parser.add_argument("--cosine1", action="store_true", help="Favors model 1's structure with details from 0", required=False)
parser.add-argument("--fine", type=str, help="Finetune the given keys on model 0", default=None, required=False)
parser.add_argument("--save_half", action="store_true", help="Save as float16", required=False)
parser.add_argument("--prune", action="store_true", help="Prune Model", required=False)
parser.add_argument("--save_safetensors", action="store_true", help="Save as .safetensors", required=False)
parser.add_argument("--keep_ema", action="store_true", help="Keep ema", required=False)
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
parser.add_argument("--functn", action="store_true", help="Add function name to the file", required=False)
parser.add_argument("--delete_source", action="store_true", help="Delete the source checkpoint file", required=False)
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)

real_mode = {"WS": "Weighted Sum",
	     "AD": "Add Difference",
	     "NoIn": "No Interpolation",
	     "TRS": "Triple Sum",
	     "ST": "Sum Twice",
	     "sAD": "smooth Add Difference",
             "TS": "Tensor Sum",
             "TD": "Train Difference",
	     "SIG": "Sigmoid Merge",
	     "GEO": "Geometric Sum",
	     "MAX": "Max Merge",
	     "RM": "Read Metadata"}
args = parser.parse_args()
device = args.device
mode = args.mode

if args.fine is not None:
    fine = [float(t) for t in args.fine.split(",")]
    fine = fineman(fine)
else:
    fine = []
if (args.cosine0 and args.cosine1) or mode != "WS":
  cosine0 = False
  cosine1 = False
else:
  cosine0 = args.cosine0
  cosine1 = args.cosine1
	
cache_filename = os.path.join(args.model_path, "cache.json")
cache_data = None

def to_half(tensor, enable):
    if enable and tensor.dtype == torch.float32:
        return tensor.half()

    return tensor

def load_weights(path, device):
  if path.endswith(".safetensors"):
      weights = safetensors.torch.load_file(path, device)
  else:
      weights = torch.load(path, device)
  weights = weights["state_dict"] if "state_dict" in weights else weights
  
  return weights

def save_weights(weights, path):
  if path.endswith(".safetensors"):
      safetensors.torch.save_file(weights, path)
  else:
      torch.save({"state_dict": weights}, path) 

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

def model_hash(filename):
    """old hash that only looks at a small part of the file and is prone to collisions"""

    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'

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

def weight_max(theta0, theta1, alpha):
    return torch.max(theta0, theta1)

def geom(theta0, theta1, alpha):
    return torch.pow(theta0, 1 - alpha) * torch.pow(theta1, alpha)

def sigmoid(theta0, theta1, alpha):
    return (1 / (1 + torch.exp(-4 * alpha))) * (theta0 + theta1) - (1 / (1 + torch.exp(-alpha))) * theta0

def weighted_sum(theta0, theta1, alpha):
  if alpha == 0:
    return theta0
  elif alpha == 1:
    return theta1
  else:
    return ((1 - alpha) * theta0) + (alpha * theta1)

def sum_twice(theta0, theta1, theta2, alpha, beta):
  if beta == 1:
    return theta2
  if alpha == 0:
    if beta == 0:
      return theta0
    else:
      return ((1 - beta) * theta0) + (beta * theta2)
  elif alpha == 1:
    if beta == 0:
      return theta1
    else:
      return ((1 - beta) * theta1) + (beta * theta2)
  else:
    if beta == 0:
      return ((1 - alpha) * theta0) + (alpha * theta1)
    else:
      return ((1 - alpha) * (1 - beta) * theta0) + (alpha * (1 - beta) * theta1) + (beta * theta2)

def triple_sum(theta0, theta1, theta2, alpha, beta):
  if alpha == 0:
    if beta == 0:
      return theta0
    elif beta == 1:
      return theta2
    else:
      return ((1 - beta) * theta0) + (beta * theta2)
  elif alpha == 1:
    if beta == 0:
      return theta1
    elif beta == 1:
      return theta1 + theta2 - theta0
    else:
      return theta1 + (beta * theta2) - (beta * theta0)
  else:
    if beta == 0:
      return ((1 - alpha) * theta0) + (alpha * theta1)
    elif beta == 1:
      return (alpha * theta1) + theta2 - (alpha * theta0)
    else:
      return ((1 - alpha - beta) * theta0) + (alpha * theta1) + (beta * theta2)

def get_difference(theta1, theta2):
    return theta1 - theta2

def add_difference(theta0, theta1_2_diff, alpha):
    return theta0 + (alpha * theta1_2_diff)

def prune_model(model):
    sd = model
    if 'state_dict' in sd:
        sd = sd['state_dict']
    sd_pruned = dict()
    for k in sd:
        cp = k.startswith('model.diffusion_model.')
        cp = cp or k.startswith('depth_model.')
        cp = cp or k.startswith('first_stage_model.')
        cp = cp or k.startswith('cond_stage_model.')
        if cp:
            k_in = k
            if args.keep_ema:
                k_ema = 'model_ema.' + k[6:].replace('.', '')
                if k_ema in sd:
                    k_in = k_ema
            if type(sd[k]) == torch.Tensor:
              if not args.save_half and sd[k].dtype in {torch.float16, torch.float64, torch.bfloat16}:
                  sd_pruned[k] = sd[k_in].to(torch.float32)
              elif args.save_half and sd[k].dtype in {torch.float32, torch.float64, torch.bfloat16}:
                  sd_pruned[k] = sd[k_in].to(torch.float16)
              else:
                  sd_pruned[k] = sd[k_in]
            else:
              sd_pruned[k] = sd[k_in]      
    return sd_pruned

output_name = args.output
if args.functn:
    if args.prune:
        output_name += "_pruned"
if args.save_safetensors:
    output_file = f'{output_name}.safetensors'
else:
    output_file = f'{output_name}.ckpt'
model_path = args.model_path
output_path = os.path.join(model_path, output_file)

fan = 0
if os.path.isfile(output_path):
  print(f"Output file already exists. Overwrite? (y/n)")
  overwrite = input()
  while overwrite != "y" and overwrite != "n":
    print("Please enter y or n")
  overwrite = input()
  if overwrite == "y":
    os.remove(output_path)
  elif overwrite == "n":
    while os.path.isfile(output_path):
      if args.save_safetensors:
        output_file = f"{output_name}_{fan}.safetensors"
      else:
        output_file = f"{output_name}_{fan}.ckpt"
      output_path = os.path.join(model_path, output_file)
      fan += 1
    print(f"Setted the file name to {output_file}\n")
	
checkpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}

checkpoint_dict_skip_on_merge = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]

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

def read_state_dict(checkpoint_file, print_global_state=False, map_location=None):
  _, extension = os.path.splitext(checkpoint_file)
  if extension.lower() == ".safetensors":
      device = map_location
      pl_sd = safetensors.torch.load_file(checkpoint_file, device=device)
  else:
      pl_sd = torch.load(checkpoint_file, map_location=map_location)

  if print_global_state and "global_step" in pl_sd:
      print(f"Global Step: {pl_sd['global_step']}")

  sd = get_state_dict_from_checkpoint(pl_sd)
  return sd

vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}

def load_vae_dict(filename, map_location):
    vae_ckpt = read_state_dict(filename, map_location=map_location)
    vae_dict_1 = {k: v for k, v in vae_ckpt.items() if k[0:4] != "loss" and k not in vae_ignore_keys}
    return vae_dict_1

model_0_path = os.path.join(args.model_path, args.model_0)
if args.model_1 is not None:
    model_1_path = os.path.join(args.model_path, args.model_1)
if args.model_2 is not None:
  model_2_path = os.path.join(args.model_path, args.model_2)

if mode in ["WS", "SIG", "GEO", "MAX","TS"]:
  interp_method = 0
  _, extension_0 = os.path.splitext(model_0_path)
  if extension_0.lower() == ".safetensors":
      model_0 = safetensors.torch.load_file(model_0_path, device=device)
  else:
      model_0 = torch.load(model_0_path, map_location=device)
  _, extension_1 = os.path.splitext(model_1_path)
  if extension_1.lower() == ".safetensors":
      model_1 = safetensors.torch.load_file(model_1_path, device=device)
  else:
      model_1 = torch.load(model_1_path, map_location=device)
		
elif mode in ["sAD", "AD", "TRS", "ST", "TD"]:
  interp_method = 0
  _, extension_0 = os.path.splitext(model_0_path)
  if extension_0.lower() == ".safetensors":
      model_0 = safetensors.torch.load_file(model_0_path, device=device)
  else:
      model_0 = torch.load(model_0_path, map_location=device)
  _, extension_1 = os.path.splitext(model_1_path)
  if extension_1.lower() == ".safetensors":
      model_1 = safetensors.torch.load_file(model_1_path, device=device)
  else:
      model_1 = torch.load(model_1_path, map_location=device)
  _, extension_2 = os.path.splitext(model_2_path)
  if extension_2.lower() == ".safetensors":
      model_2 = safetensors.torch.load_file(model_2_path, device=device)
  else:
      model_2 = torch.load(model_2_path, map_location=device)

elif mode == "NoIn":
  interp_method = 2
  _, extension_0 = os.path.splitext(model_0_path)
  if extension_0.lower() == ".safetensors":
      model_0 = safetensors.torch.load_file(model_0_path, device=device)
  else:
      model_0 = torch.load(model_0_path, map_location=device)
		
elif mode == "RM":
  print(sha256(model_0_path, f"checkpoint/{os.path.splitext(os.path.basename(model_0_path))[0]}"))
  meta = read_metadata_from_safetensors(model_0_path)
  print(json.dumps(meta, indent=2))
  with open("./" + output_name + ".json", mode="a+") as dmp:
    json.dump(meta, dmp, indent=4)
  exit()
	
if args.vae is not None:
    _, extension_vae = os.path.splitext(args.vae)
    vae_name = os.path.splitext(os.path.basename(args.vae))[0]
    if extension_vae.lower() == ".safetensors":
      vae = safetensors.torch.load_file(args.vae, device=device)
    else:
      vae = torch.load(args.vae, map_location=device)

def rinfo(string, seed):
    tram = string.split("[")
    string = tram[0]
    try:
      fe = tram[1].replace("]","")
    except:
      fe = None
    parsed = [a for a in string.replace("\n",",").replace(","," ").split(" ") if (a != "" and a != " ")]
    if type(parsed) is list:
        try:
            rmin = float(parsed[0])
        except ValueError: rmin = 0.0
        try:
            try:
                rmax = float(parsed[1])
            except ValueError: rmax = 1.0
        except IndexError: rmax = 1.0
        try:
            try:
                get = int(parsed[2])
            except ValueError: get = seed
        except IndexError: get = seed
    else:
        rmin = 0.0
        rmax = 1.0
        get = seed
    return f"({rmin},{rmax},{get},[{fe}])"

def roundeep(term):
    deep = term
    if len(deep) > 0:
        for d in deep:
            dbs,dws,dr = d.split(":")[0],d.split(":")[1],d.split(":")[2]
            dr = round(float(dr),3)
            d = f"{dbs}:{dws}:{dr}"
    else:
        return None
    return deep
            
alpha_seed = None
beta_seed = None
if args.rand_alpha is not None:
    alphas, alpha_seed, deep_a = rand_ratio(args.rand_alpha)
    alpha_info = rinfo(args.rand_alpha, alpha_seed)
    args.alpha = wgta(alphas)
if args.rand_beta is not None:
    betas, beta_seed, deep_b = rand_ratio(args.rand_beta)
    beta_info = rinfo(args.rand_beta, beta_seed)
    args.beta = wgtb(betas)
	
usebeta = False	
if type(args.alpha) == list:
    weights_a = args.alpha
    alpha = weights_a.pop(0)
    round_a = [round(a, 3) for a in weights_a]
    round_deep_a = roundeep(deep_a)
    if args.rand_alpha is not None:
        alpha_info = f"preset:[{alpha_info}],{round(alpha, 3)},[{round_a},[{round_deep_a}]]"
    else:
        alpha_info = f"{round(alpha,3)},[{round_a},[{round_deep_a}]]"
else:
  weights_a = None
  round_a = None
  alpha = args.alpha
  alpha_info = f"{round(args.alpha,3)}"
	
if mode in ["TRS","ST","TS"]:
  usebeta = True
  if type(args.beta) == list:
    weights_b = args.beta
    beta = weights_b.pop(0)
    round_b = [round(b, 3) for b in weights_b]
    round_deep_b = roundeep(deep_b)
    if args.rand_beta is not None:
        beta_info = f"preset:[{beta_info}],{round(beta,3)},[{round_b},[{round_deep_b}]]"
    else:
        beta_info = f"{round(beta,3)},[{round_b},[{round_deep_b}]]"
  else:
    weights_b = None
    round_b = None
    beta = args.beta
    beta_info = f"{round(args.beta,3)}"
else:
  weights_b = None
  round_b = None
  beta = None
  beta_info = None
	
model_0_name = args.m0_name if args.m0_name is not None else os.path.splitext(os.path.basename(model_0_path))[0]
model_0_bname = os.path.splitext(os.path.basename(model_0_path))[0]
model_0_sha256 = sha256_from_cache(model_0_path, f"checkpoint/{model_0_bname}")
if mode != "NoIn":
  model_1_name = args.m1_name if args.m1_name is not None else os.path.splitext(os.path.basename(model_1_path))[0]
  model_1_bname = os.path.splitext(os.path.basename(model_1_path))[0]
  model_1_sha256 = sha256_from_cache(model_1_path, f"checkpoint/{model_1_bname}")
if mode in ["sAD", "AD", "TRS", "ST", "TD"]:
  model_2_name = args.m2_name if args.m2_name is not None else os.path.splitext(os.path.basename(model_2_path))[0]
  model_2_bname = os.path.splitext(os.path.basename(model_2_path))[0]
  model_2_sha256 = sha256_from_cache(model_2_path, f"checkpoint/{model_2_bname}")
if args.prune:
  model_0 = prune_model(model_0)
  if mode != "NoIn":
    model_1 = prune_model(model_1)
  if mode in ["sAD", "AD", "TRS", "ST"]:
    model_2 = prune_model(model_2)
if args.vae is not None:
  vae_name = os.path.splitext(os.path.basename(args.vae))[0]

metadata = {"format": "pt", "sd_merge_models": {}, "sd_merge_recipe": None}

calculate = []
if cosine0:
  calculate.append("cosine_0")
if cosine1:
  calculate.append("cosine_1")
if args.use_dif_10:
    calculate.append("use_dif_10")
if args.use_dif_20:
    calculate.append("use_dif_20")
if args.use_dif_21:
    calculate.append("use_dif_21")
if args.fine is not None:
    calculate.append(f"fine[{fine}]")
calcl = ",".join(calculate) if calculate != [] else None

merge_recipe = {
"type": "merge-models-chattiori", # indicate this model was merged with chattiori's model mereger
"primary_model_hash": sha256_from_cache(model_0_path, f"checkpoint/{model_0_bname}"),
"secondary_model_hash": sha256_from_cache(model_1_path, f"checkpoint/{model_1_bname}") if mode != "NoIn" else None,
"tertiary_model_hash": sha256_from_cache(model_2_path, f"checkpoint/{model_2_bname}") if mode in ["sAD", "AD", "TRS", "ST","TD"] else None,
"merge_method": real_mode[mode],
"block_weights": (weights_a is not None or weights_b is not None),
"alpha_info": alpha_info,
"beta_info": beta_info,
"calculation": calcl,
"save_as_half": args.save_half,
"output_name": output_name,
"bake_in_vae": vae_name if args.vae is not None else False,
"pruned": args.prune
}
metadata["sd_merge_recipe"] = json.dumps(merge_recipe)

def add_model_metadata(filename, model_name):
  sha256_t = sha256(filename, f"checkpoint/{os.path.splitext(os.path.basename(filename))[0]}")
  hash_t = model_hash(filename)
  _, extension_t = os.path.splitext(filename)
  if extension_t.lower() == ".safetensors":
    metadata_t = read_metadata_from_safetensors(filename)
  else:
    metadata_t = {}
  metadata["sd_merge_models"][sha256_t] = {
  "name": model_name,
  "legacy_hash": hash_t,
  "sd_merge_recipe": metadata_t.get("sd_merge_recipe", None)
  }

  metadata["sd_merge_models"].update(metadata_t.get("sd_merge_models", {}))

add_model_metadata(model_0_path, model_0_name)
if mode != "NoIn":
  add_model_metadata(model_1_path, model_1_name)
if mode in ["sAD", "AD", "TRS", "ST","TD"]:
  add_model_metadata(model_2_path, model_2_name)

metadata["sd_merge_models"] = json.dumps(metadata["sd_merge_models"])

def filename_weighted_sum():
  a = model_0_name
  b = model_1_name
  Ma = round(1 - alpha, 2)
  Mb = round(alpha, 2)

  return f"{Ma}({a}) + {Mb}({b}) LIN"

def filename_geom():
  a = model_0_name
  b = model_1_name
  Ma = round(1 - alpha, 2)
  Mb = round(alpha, 2)

  return f"{Ma}({a}) + {Mb}({b}) GEO"

def filename_sum_twice():
  a = model_0_name
  b = model_1_name
  c = model_2_name
  Ma = round(1 - alpha, 2)
  Mb = round(alpha, 2)
  Mab = round(1 - beta, 2)
  Mc = round(beta, 2)

  return f"{Mab}({Ma}({a}) + {Mb}({b})) + {Mc}({c})"

def filename_triple_sum():
  a = model_0_name
  b = model_1_name
  c = model_2_name
  Ma = round(1 - alpha - beta, 2)
  Mb = round(alpha, 2)
  Mc = round(beta, 2)

  return f"{Ma}({a}) + {Mb}({b}) + {Mc}({c})"

def filename_max():
  a = model_0_name
  b = model_1_name
  Ma = round(1 - alpha, 2)
  Mb = round(alpha, 2)

  return f"{a} + {b} MAX"

def filename_sigmoid():
  a = model_0_name
  b = model_1_name
  Ma = round(1 - alpha, 2)
  Mb = round(alpha, 2)

  return f"{Ma}({a}) + {Mb}({b}) SIG"

def filename_add_difference():
  a = model_0_name
  b = model_1_name
  c = model_2_name
  M = round(alpha, 2)

  return f"{a} + {M}({b} - {c})"

def filename_nothing():
  return model_0_name

def blocker_S(blocks):
    blocks = blocks.split(" ")
    output = ""
    for w in blocks:
        flagger=[False]*26
        changer = True
        if "-" in w:
            wt = [wt.strip() for wt in w.split('-')]
            if  blockid.index(wt[1]) > blockid.index(wt[0]):
                flagger[blockid.index(wt[0]):blockid.index(wt[1])+1] = [changer]*(blockid.index(wt[1])-blockid.index(wt[0])+1)
            else:
                flagger[blockid.index(wt[1]):blockid.index(wt[0])+1] = [changer]*(blockid.index(wt[0])-blockid.index(wt[1])+1)
        else:
            output = output + " " + w if output != "" else w
            return output
        for i in range(26):
            if flagger[i]: output = output + " " + blockid[i] if output !="" else blockid[i]
        return output
    
theta_funcs = {
    "WS":   (filename_weighted_sum, None, weighted_sum),
    "AD":   (filename_add_difference, get_difference, add_difference),
    "sAD":   (filename_add_difference, get_difference, add_difference),
    "TD":   (filename_add_difference, None, add_difference),
    "TS":   (filename_weighted_sum, None, weighted_sum),
    "TRS":  (filename_triple_sum, None, triple_sum),
    "ST":   (filename_sum_twice, None, sum_twice),
    "NoIn": (filename_nothing, None, None),
    "SIG":  (filename_sigmoid, None, sigmoid),
    "GEO":  (filename_geom, None, geom),
    "MAX":  (filename_max, None, weight_max),
}
filename_generator, theta_func1, theta_func2 = theta_funcs[mode] 

if mode in ["sAD", "AD", "TRS", "ST", "TD"]:
  print(f"Loading {model_2_name}...")
  theta_2 = read_state_dict(model_2_path, map_location=device)

if theta_func2:
  print(f"Loading {model_1_name}...")
  theta_1 = read_state_dict(model_1_path, map_location=device)
else:
  theta_1 = None

if theta_func1:
  for key in tqdm(theta_1.keys(), desc="Getting Difference of Model 1 and 2"):
    if 'model' in key:
      if key in theta_2:
          t2 = theta_2.get(key, torch.zeros_like(theta_1[key]))
          theta_1[key] = theta_func1(theta_1[key], t2)
      else:
          theta_1[key] = torch.zeros_like(theta_1[key])
  del theta_2

print(f"Loading {model_0_name}...")
if mode == "TS":
    theta_t = read_state_dict(model_0_path, map_location=device)
    theta_0 ={}
    for key in theta_t:
        theta_0[key] = theta_t[key].clone()
    del theta_t
else:
    theta_0 = read_state_dict(model_0_path, map_location=device)

if args.use_dif_21:
    theta_3 = copy.deepcopy(theta_1)
    for key in tqdm(theta_2.keys(), desc="Getting Difference of Model 1 and 2"):
        if 'model' in key:
          if key in theta_3:
              t2 = theta_3.get(key, torch.zeros_like(theta_2[key]))
              theta_2[key] = theta_func1(theta_2[key], t2)
          else:
              theta_2[key] = torch.zeros_like(theta_2[key])
    del theta_3

re_inp = re.compile(r'\.input_blocks\.(\d+)\.')  # 12
re_mid = re.compile(r'\.middle_block\.(\d+)\.')  # 1
re_out = re.compile(r'\.output_blocks\.(\d+)\.') # 12

if args.use_dif_10:
    theta_3 = copy.deepcopy(theta_0)
    for key in tqdm(theta_1.keys(), desc="Getting Difference of Model 0 and 1"):
        if 'model' in key:
          if key in theta_3:
              t2 = theta_3.get(key, torch.zeros_like(theta_1[key]))
              theta_1[key] = theta_func1(theta_1[key], t2)
          else:
              theta_1[key] = torch.zeros_like(theta_1[key])
    del theta_3

if args.use_dif_20:
    theta_3 = copy.deepcopy(theta_0)
    for key in tqdm(theta_2.keys(), desc="Getting Difference of Model 0 and 2"):
        if 'model' in key:
          if key in theta_3:
              t2 = theta_3.get(key, torch.zeros_like(theta_2[key]))
              theta_2[key] = theta_func1(theta_2[key], t2)
          else:
              theta_2[key] = torch.zeros_like(theta_2[key])
    del theta_3

    
if cosine0: #favors modelA's structure with details from B
    sim = torch.nn.CosineSimilarity(dim=0)
    sims = np.array([], dtype=np.float64)
    for key in (tqdm(theta_0.keys(), desc="Caluculating Cosine 0")):
        # skip VAE model parameters to get better results
        if "first_stage_model" in key: continue
        if "model" in key and key in theta_1:
            theta_0_norm = nn.functional.normalize(theta_0[key].to(torch.float32), p=2, dim=0)
            theta_1_norm = nn.functional.normalize(theta_1[key].to(torch.float32), p=2, dim=0)
            simab = sim(theta_0_norm, theta_1_norm)
            sims = np.append(sims,simab.numpy())
    sims = sims[~np.isnan(sims)]
    sims = np.delete(sims, np.where(sims<np.percentile(sims, 1 ,method = 'midpoint')))
    sims = np.delete(sims, np.where(sims>np.percentile(sims, 99 ,method = 'midpoint')))

if cosine1: #favors modelB's structure with details from A
    sim = torch.nn.CosineSimilarity(dim=0)
    sims = np.array([], dtype=np.float64)
    for key in (tqdm(theta_0.keys(), desc="Caluculating Cosine 1")):
        # skip VAE model parameters to get better results
        if "first_stage_model" in key: continue
        if "model" in key and key in theta_1:
            simab = sim(theta_0[key].to(torch.float32), theta_1[key].to(torch.float32))
            dot_product = torch.dot(theta_0[key].view(-1).to(torch.float32), theta_1[key].view(-1).to(torch.float32))
            magnitude_similarity = dot_product / (torch.norm(theta_0[key].to(torch.float32)) * torch.norm(theta_1[key].to(torch.float32)))
            combined_similarity = (simab + magnitude_similarity) / 2.0
            sims = np.append(sims, combined_similarity.numpy())
    sims = sims[~np.isnan(sims)]
    sims = np.delete(sims, np.where(sims < np.percentile(sims, 1, method='midpoint')))
    sims = np.delete(sims, np.where(sims > np.percentile(sims, 99, method='midpoint')))

if mode != "NoIn":
  for key in tqdm(theta_0.keys(), desc="Merging..."):
    if theta_1 and "model" in key and key in theta_1:    
      if (usebeta or mode == "TD") and not key in theta_2:
         continue
      weight_index = -1
      current_alpha = alpha
      current_beta = beta
      if key in checkpoint_dict_skip_on_merge:
        continue
      a = theta_0[key]
      b = theta_1[key]
      if usebeta:
        c = theta_2[key]
      # check weighted and U-Net or not
      if (weights_a is not None or weights_b is not None) and 'model.diffusion_model.' in key:
        # check block index
        weight_index = -1

        if 'time_embed' in key:
            weight_index = 0                # before input blocks
        elif '.out.' in key:
            weight_index = NUM_TOTAL_BLOCKS - 1     # after output blocks
        else:
          m = re_inp.search(key)
          if m:
            inp_idx = int(m.groups()[0])
            weight_index = inp_idx
          else:
            m = re_mid.search(key)
          if m:
              weight_index = NUM_INPUT_BLOCKS
          else:
              m = re_out.search(key)
              if m:
                out_idx = int(m.groups()[0])
                weight_index = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + out_idx

        if weight_index >= NUM_TOTAL_BLOCKS:
            print(f"ERROR: illegal block index: {key}")

        if weight_index >= 0:
            if weights_a is not None:
              current_alpha = weights_a[weight_index]
            if usebeta:
              if weights_b is not None:
                  current_beta = weights_b[weight_index]
			
      if len(deep_a) > 0:
        skey = key + blockid[weight_index+1]
        for d in deep_a:
          if d.count(":") != 2 :continue
          dbs,dws,dr = d.split(":")[0],d.split(":")[1],d.split(":")[2]
          dbs = blocker_S(dbs)
          dbs,dws = dbs.split(" "), dws.split(" ")
          dbn,dbs = (True,dbs[1:]) if dbs[0] == "NOT" else (False,dbs)
          dwn,dws = (True,dws[1:]) if dws[0] == "NOT" else (False,dws)
          flag = dbn
          for db in dbs:
            if db in skey:
              flag = not dbn
          if flag:flag = dwn
          else:continue
          for dw in dws:
            if dw in skey:
              flag = not dwn
          if flag:
            dr = float(dr)
            current_alpha = dr

      if len(deep_b) > 0:
        skey = key + blockid[weight_index+1]
        for d in deep_b:
          if d.count(":") != 2 :continue
          dbs,dws,dr = d.split(":")[0],d.split(":")[1],d.split(":")[2]
          dbs,dws = dbs.split(" "), dws.split(" ")
          dbs = blocker_S(dbs)
          dbn,dbs = (True,dbs[1:]) if dbs[0] == "NOT" else (False,dbs)
          dwn,dws = (True,dws[1:]) if dws[0] == "NOT" else (False,dws)
          flag = dbn
          for db in dbs:
            if db in skey:
              flag = not dbn
          if flag:flag = dwn
          else:continue
          for dw in dws:
            if dw in skey:
              flag = not dwn
          if flag:
            dr = float(dr)
            current_beta = dr
			
      # this enables merging an inpainting model (A) with another one (B);
      # where normal model would have 4 channels, for latenst space, inpainting model would
      # have another 4 channels for unmasked picture's latent space, plus one channel for mask, for a total of 9
      if cosine0:
        # skip VAE model parameters to get better results
        if "first_stage_model" in key: continue
        if "model" in key and key in theta_0:
            # Normalize the vectors before merging
            theta_0_norm = nn.functional.normalize(a.to(torch.float32), p=2, dim=0)
            theta_1_norm = nn.functional.normalize(b.to(torch.float32), p=2, dim=0)
            simab = sim(theta_0_norm, theta_1_norm)
            dot_product = torch.dot(theta_0_norm.view(-1), theta_1_norm.view(-1))
            magnitude_similarity = dot_product / (torch.norm(theta_0_norm) * torch.norm(theta_1_norm))
            combined_similarity = (simab + magnitude_similarity) / 2.0
            k = (combined_similarity - sims.min()) / (sims.max() - sims.min())
            k = k - abs(current_alpha)
            k = k.clip(min=0,max=1.0)
            theta_0[key] = b * (1 - k) + a * k
	
      elif cosine1:
        # skip VAE model parameters to get better results
        if "first_stage_model" in key: continue
        if "model" in key and key in theta_0:
            simab = sim(a.to(torch.float32), b.to(torch.float32))
            dot_product = torch.dot(a.view(-1).to(torch.float32), b.view(-1).to(torch.float32))
            magnitude_similarity = dot_product / (torch.norm(a.to(torch.float32)) * torch.norm(b.to(torch.float32)))
            combined_similarity = (simab + magnitude_similarity) / 2.0
            k = (combined_similarity - sims.min()) / (sims.max() - sims.min())
            k = k - current_alpha
            k = k.clip(min=0,max=1.0)
            theta_0[key] = b * (1 - k) + a * k
		
      elif mode == "sAD":
        # Apply median filter to the weight differences
        filtered_diff = scipy.ndimage.median_filter(b.to(torch.float32).cpu().numpy(), size=3)
        # Apply Gaussian filter to the filtered differences
        filtered_diff = scipy.ndimage.gaussian_filter(filtered_diff, sigma=1)
        b = torch.tensor(filtered_diff)
        # Add the filtered differences to the original weights
        theta_0[key] = a + current_alpha * b
        
      elif mode == "TD":
        # Check if theta_1[key] is equal to theta_2[key]
        if torch.allclose(theta_1[key].float(), theta_2[key].float(), rtol=0, atol=0):
            theta_2[key] = theta_0[key]
            continue

        diff_AB = theta_1[key].float() - theta_2[key].float()

        distance_A0 = torch.abs(theta_1[key].float() - theta_2[key].float())
        distance_A1 = torch.abs(theta_1[key].float() - theta_0[key].float())

        sum_distances = distance_A0 + distance_A1

        scale = torch.where(sum_distances != 0, distance_A1 / sum_distances, torch.tensor(0.).float())
        sign_scale = torch.sign(theta_1[key].float() - theta_2[key].float())
        scale = sign_scale * torch.abs(scale)

        new_diff = scale * torch.abs(diff_AB)
        theta_0[key] = theta_0[key] + (new_diff * (current_alpha*1.8))
      elif mode == "TS":
            dim = theta_0[key].dim()
            if dim == 0 : continue
            if current_alpha+current_beta <= 1 :
                talphas = int(theta_0[key].shape[0]*(current_beta))
                talphae = int(theta_0[key].shape[0]*(current_alpha+current_beta))
                if dim == 1:
                    theta_0[key][talphas:talphae] = theta_1[key][talphas:talphae].clone()

                elif dim == 2:
                    theta_0[key][talphas:talphae,:] = theta_1[key][talphas:talphae,:].clone()

                elif dim == 3:
                    theta_0[key][talphas:talphae,:,:] = theta_1[key][talphas:talphae,:,:].clone()

                elif dim == 4:
                    theta_0[key][talphas:talphae,:,:,:] = theta_1[key][talphas:talphae,:,:,:].clone()

            else:
                talphas = int(theta_0[key].shape[0]*(current_alpha+current_beta-1))
                talphae = int(theta_0[key].shape[0]*(current_beta))
                theta_t = theta_1[key].clone()
                if dim == 1:
                    theta_t[talphas:talphae] = theta_0[key][talphas:talphae].clone()

                elif dim == 2:
                    theta_t[talphas:talphae,:] = theta_0[key][talphas:talphae,:].clone()

                elif dim == 3:
                    theta_t[talphas:talphae,:,:] = theta_0[key][talphas:talphae,:,:].clone()

                elif dim == 4:
                    theta_t[talphas:talphae,:,:,:] = theta_0[key][talphas:talphae,:,:,:].clone()
                theta_0[key] = theta_t
      else:
        if a.shape != b.shape and a.shape[0:1] + a.shape[2:] == b.shape[0:1] + b.shape[2:]:
          if a.shape[1] == 4 and b.shape[1] == 9:
            raise RuntimeError("When merging inpainting model with a normal one, A must be the inpainting model.")
          if a.shape[1] == 4 and b.shape[1] == 8:
            raise RuntimeError("When merging instruct-pix2pix model with a normal one, A must be the instruct-pix2pix model.")

          if a.shape[1] == 8 and b.shape[1] == 4:#If we have an Instruct-Pix2Pix model...
            if usebeta:
              theta_0[key][:, 0:4, :, :] = theta_func2(a[:, 0:4, :, :], b, c, current_alpha, current_beta)
            else:
              theta_0[key][:, 0:4, :, :] = theta_func2(a[:, 0:4, :, :], b, current_alpha)
            result_is_instruct_pix2pix_model = True
          else:
            assert a.shape[1] == 9 and b.shape[1] == 4, f"Bad dimensions for merged layer {key}: A={a.shape}, B={b.shape}"
            if usebeta:
              theta_0[key][:, 0:4, :, :] = theta_func2(a[:, 0:4, :, :], b, c, current_alpha, current_beta)
            else:
              theta_0[key][:, 0:4, :, :] = theta_func2(a[:, 0:4, :, :], b, current_alpha)
            result_is_inpainting_model = True
        else:
          if usebeta:
            theta_0[key] = theta_func2(a, b, c, current_alpha, current_beta)
          else:
            theta_0[key] = theta_func2(a, b, current_alpha)

      if any(item in key for item in FINETUNES) and fine:
        index = FINETUNES.index(key)
        print(key,fine[index])
        if 5 > index : 
            theta_0[key] =theta_0[key]* fine[index] 
        else :theta_0[key] =theta_0[key] + torch.tensor(fine[5])
        
      theta_0[key] = to_half(theta_0[key], args.save_half)
  for key in tqdm(theta_1.keys(), desc="Remerging..."):
        if key in checkpoint_dict_skip_on_merge:
            continue
        if "model" in key and key not in theta_0:
            theta_0.update({key:theta_1[key]})
  del theta_1
  try:
    if theta_2:
        del theta_2
  except NameError:
    pass
            
if args.vae is not None:
    print(f"Baking in VAE")
    vae_dict = load_vae_dict(args.vae, map_location=device)
    for key in vae_dict.keys():
        theta_0_key = 'first_stage_model.' + key
        if theta_0_key in theta_0:
            theta_0[theta_0_key] = to_half(vae_dict[key], args.save_half)
    del vae_dict
    
if args.save_half and not theta_func2:
    for key in theta_0.keys():
        theta_0[key] = to_half(theta_0[key], args.save_half)   

loaded = None
# check if output file already exists, ask to overwrite
if args.prune:
  print("Making Full-sized File...\n")
  output_a = os.path.join(model_path, "test.safetensors")
  if os.path.isfile(output_a):
    os.remove(output_a)
  safetensors.torch.save_file(theta_0, output_a,metadata=metadata)
  sd = safetensors.torch.load_file(output_a, device=device)
  model = prune_model(sd)
  file_size_temp = round(os.path.getsize(output_a) / 1073741824,2)
  print(f"Pruning {output_file}({file_size_temp}G)...")
  if args.save_safetensors:
    with torch.no_grad():
        safetensors.torch.save_file(model, output_path, metadata=metadata)
  else:
      torch.save({"state_dict": model}, output_path)
  del model
  os.remove(output_a)
else:
  print(f"Saving as {output_file}...")
  if args.save_safetensors:
    with torch.no_grad():
        safetensors.torch.save_file(theta_0, output_path, metadata=metadata)
  else:
      torch.save({"state_dict": theta_0}, output_path)
if args.delete_source:
    os.remove(model_0_path)
    if mode != "NoIn":
      os.remove(model_1_path)
    if mode in ["sAD", "AD", "TRS", "ST"]:
      os.remove(model_2_path)
del theta_0
file_size = round(os.path.getsize(output_path) / 1073741824,2)
print(f"Done! ({file_size}G)")
