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
from tq import tqdm

FINETUNEX = ["IN","OUT","OUT2","CONT","BRI","COL1","COL2","COL3"]
NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS
BLOCKID=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKIDXLL=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","VAE"]
BLOCKIDXL=['BASE', 'IN0', 'IN1', 'IN2', 'IN3', 'IN4', 'IN5', 'IN6', 'IN7', 'IN8', 'M', 'OUT0', 'OUT1', 'OUT2', 'OUT3', 'OUT4', 'OUT5', 'OUT6', 'OUT7', 'OUT8', 'VAE']

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
def wgt(string, dp):
    if type(string) == int or type(string) == float:
        return float(string), dp
    elif type(string) == list:
        useblocks = True
        string, dp = deepblock(string)
        while type(string) == list and len(string) == 1:
          string = string[0]
        return string, dp
    elif type(string) == str:
        useblocks = True
        string, dp = deepblock([string])
        while type(string) == list and len(string) == 1:
          string = string[0]
        return string, dp

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
                        dr1 = ratios[BLOCKID.index(db)]
                        dr = dr1 * (1 - drat) + dr0 * drat
                        ratios[BLOCKID.index(db)] = dr
                else:
                    for db in dbs:
                        ratios[BLOCKID.index(db)] = float(dr)
            else:
                if "(" in dr:
                    dr0, drat = float(dr.split("(")[0]), dr.split("(")[1].replace(")","")
                    for db in dbs:
                        dr1 = ratios[BLOCKID.index(db)]
                        dr = dr1 * (1 - drat) + dr0 * drat
                        deep_res.append(f"{db}:{dws}:{dr}")
                else:
                    dr0 = float(dr)
                    for db in dbs:
                        deep_res.append(f"{db}:{dws}:{dr0}")
    info = rinfo(string, seed)
    base_ratio, deep_res = wgt(ratios, deep_res)
    return ratios, seed, deep_res, info, base_ratio

COLS = [[-1,1/3,2/3],[1,1,0],[0,-1,-1],[1,0,1]]
COLSXL = [[0,0,1],[1,0,0],[-1,-1,0],[-1,1,0]]

def colorcalc(cols,isxl):
    colors = COLSXL if isxl else COLS
    outs = [[y * cols[i] * 0.02 for y in x] for i,x in enumerate(colors)]
    return [sum(x) for x in zip(*outs)]

def fineman(fine,isxl):
    r = [
        1 - fine[0] * 0.01,
        1+ fine[0] * 0.02,
        1 - fine[1] * 0.01,
        1+ fine[1] * 0.02,
        1 - fine[2] * 0.01,
        [fine[3]*0.02] + colorcalc(fine[4:8],isxl)
                ]
    return r

def weighttoxl(weight):
    weight = weight[:9] + weight[12:22] +[0]
    return weight

def parse_ratio(ratios, info, dp):
    if type(ratios) == list:
        weights = ratios
        ratio = weights.pop(0)
        rounded = [round(a, 3) for a in weights]
        round_deep = roundeep(dp)
        if info != "":
            info = f"preset:[{info}],{round(ratio, 3)},[{rounded},[{round_deep}]]"
        else:
            info = f"{round(ratio,3)},[{rounded},[{round_deep}]]"
    else:
      weights = None
      ratio = ratios
      info = f"{round(ratios,3)}"
    return weights, ratio, info

FINETUNES = [
"model.diffusion_model.input_blocks.0.0.weight",
"model.diffusion_model.input_blocks.0.0.bias",
"model.diffusion_model.out.0.weight",
"model.diffusion_model.out.0.bias",
"model.diffusion_model.out.2.weight",
"model.diffusion_model.out.2.bias",
]

parser = argparse.ArgumentParser(description="Merge two or three models")
parser.add_argument("mode", choices=["DARE","WS","AD","NoIn","MD","SIM","TRS","ST","sAD","TD","TS","SIG","GEO","MAX","RM"], help="Merging mode")
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
parser.add_argument("--alpha", help="Alpha value, optional, defaults to 0", default=0.0, required=False)
parser.add_argument("--rand_alpha", type=str, help="Random Alpha value, optional", default=None, required=False)
parser.add_argument("--beta", help="Beta value, optional, defaults to 0", default=0.0, required=False)
parser.add_argument("--rand_beta", type=str, help="Random Beta value, optional", default=None, required=False)
parser.add_argument("--cosine0", action="store_true", help="Favors model 0's structure with details from 1", required=False)
parser.add_argument("--cosine1", action="store_true", help="Favors model 1's structure with details from 0", required=False)
parser.add_argument("--fine", type=str, help="Finetune the given keys on model 0", default=None, required=False)
parser.add_argument("--save_half", action="store_true", help="Save as float16", required=False)
parser.add_argument("--prune", action="store_true", help="Prune Model", required=False)
parser.add_argument("--save_quarter", action="store_true", help="Save as float8", required=False)
parser.add_argument("--save_safetensors", action="store_true", help="Save as .safetensors", required=False)
parser.add_argument("--keep_ema", action="store_true", help="Keep ema", required=False)
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
parser.add_argument("--functn", action="store_true", help="Add function name to the file", required=False)
parser.add_argument("--delete_source", action="store_true", help="Delete the source checkpoint file", required=False)
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
parser.add_argument("--no_metadata", action="store_true", help="Save without metadata", required=False)

real_mode = {"WS": "Weighted Sum",
	     "AD": "Add Difference",
	     "NoIn": "No Interpolation",
             "MD": "Multiply Difference",
             "SIM": "Similarity Add Difference",
	     "TRS": "Triple Sum",
	     "ST": "Sum Twice",
	     "sAD": "smooth Add Difference",
             "TS": "Tensor Sum",
             "TD": "Train Difference",
	     "SIG": "Sigmoid Merge",
	     "GEO": "Geometric Sum",
	     "MAX": "Max Merge",
	     "RM": "Read Metadata",
             "DARE":"DARE Merge"}

args = parser.parse_args()
device = args.device
mode = args.mode
args.alpha, deep_a = wgt(args.alpha, deep_a)
args.beta, deep_b = wgt(args.beta, deep_b)
if (args.cosine0 and args.cosine1) or mode != "WS":
  cosine0 = False
  cosine1 = False
else:
  cosine0 = args.cosine0
  cosine1 = args.cosine1
	
cache_filename = os.path.join(args.model_path, "cache.json")
cache_data = None

def to_half(tensor, enable):
    if enable and tensor.dtype in {torch.float32, torch.float64, torch.bfloat16}:
        return tensor.half()

    return tensor

def to_half_k(sd, enable):
    if enable:
      for key in sd.keys():
          if 'model' in key and sd[key].dtype in {torch.float32, torch.float64, torch.bfloat16}:
              sd[key] = sd[key].half()
    return sd

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
    
def sha256(filename, title):
    hashes = cache("hashes")

    sha256_value = sha256_from_cache(filename, title, 1)
    if sha256_value is not None:
        return sha256_value

    print(f"Calculating sha256 for {filename}: ", end='')
    sha256_value = calculate_sha256(filename)
    hash_value = model_hash(filename)
    print(f"{sha256_value}")

    hashes[title] = {
        "mtime": os.path.getmtime(filename),
        "sha256": sha256_value,
        "model_hash": hash_value,
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
        sha256(filename, title)
      else:
        return None

    cached_sha256 = hashes[title].get("sha256", None)
    cached_hash = hashes[title].get("model_hash",None)
    cached_mtime = hashes[title].get("mtime", 0)

    if cached_sha256 is None:
        return None

    return cached_sha256, cached_hash

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

def multiply_difference(theta0, theta1, theta2, alpha, beta):
    diff_a = torch.pow(torch.abs(theta0.float() - theta2), (1 - alpha))
    diff_b = torch.pow(torch.abs(theta1.float() - theta2), alpha)
    difference = torch.copysign(diff_a * diff_b, weighted_sum(theta0, theta1, beta) - theta2)
    return theta2 + difference.to(theta2.dtype)

def similarity_add_difference(a, b, c, alpha, beta):
    threshold = torch.maximum(torch.abs(a), torch.abs(b))
    similarity = ((a * b / threshold**2) + 1) / 2
    similarity = torch.nan_to_num(similarity * beta, nan=beta)

    ab_diff = a + alpha * (b - c)
    ab_sum = (1 - alpha / 2) * a + (alpha / 2) * b
    return (1 - similarity) * ab_diff + similarity * ab_sum

def dare_merge(theta0, theta1, alpha):
    m = torch.bernoulli(torch.full_like(input=theta0.float(), fill_value=0.5), generator=rand_generator)
    alpha = torch.mul(m, alpha / 0.5)
    return torch.lerp(theta0.float(), theta1.float(), alpha).to(theta0.dtype)

def prune_model(theta, name, isxl=False):
    sd_pruned = dict()
    pr = tqdm(theta.keys(), desc=f"Pruning {name}...")
    for key in pr:
        cp = key.startswith('model.diffusion_model.') or key.startswith('depth_model.') or key.startswith('first_stage_model.') or key.startswith("conditioner." if isxl else 'cond_stage_model.')
        if cp:
            k_in = key
            if args.keep_ema:
                k_ema = 'model_ema.' + key[6:].replace('.', '')
                if k_ema in theta:
                    k_in = k_ema
            if type(theta[key]) == torch.Tensor:
              if args.save_quarter and theta[key].dtype in {torch.float32, torch.float16, torch.float64, torch.bfloat16}:
                  sd_pruned[key] = theta[k_in].to(torch.float8_e4m3fn)
              elif not args.save_half and theta[key].dtype in {torch.float16, torch.float64, torch.bfloat16}:
                  sd_pruned[key] = theta[k_in].to(torch.float32)
              elif args.save_half and theta[key].dtype in {torch.float32, torch.float64, torch.bfloat16}:
                  sd_pruned[key] = theta[k_in].to(torch.float16)
              else:
                  sd_pruned[key] = theta[k_in]
            else:
              sd_pruned[key] = theta[k_in]      
    return sd_pruned

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

vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}

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

def load_model(path, device, sha256=True):
    if path.endswith(".safetensors"):
        weights = safetensors.torch.load_file(path, device)
        metadata = read_metadata_from_safetensors(path)
    else:
        weights = torch.load(path, device)
        metadata = {}
    bname = os.path.splitext(os.path.basename(path))[0]
    if sha256:
        s256, hashed = sha256_from_cache(path, f"checkpoint/{bname}")
    else:
        s256 = None
        hashed = None
        metadata = None
    weights = get_state_dict_from_checkpoint(weights)
    return weights, s256, hashed, metadata

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

alpha_seed = None
beta_seed = None
usebeta = False
alpha_info = ""
beta_info = ""
weights_a = None
weights_b = None
beta = None
if args.rand_alpha is not None:
    alphas, alpha_seed, deep_a, alpha_info, args.alpha = rand_ratio(args.rand_alpha)
if args.rand_beta is not None:
    betas, beta_seed, deep_b, beta_info, args.beta = rand_ratio(args.rand_beta)

theta_0 = None
theta_1 = None
theta_2 = None
model_0_sha256 = None
model_1_sha256 = None
model_2_sha256 = None
model_0_path = os.path.join(args.model_path, args.model_0)
if mode != "RM":
    interp_method = 2
    model_0_name = args.m0_name if args.m0_name is not None else os.path.splitext(os.path.basename(model_0_path))[0]
    print(f"Loading {model_0_name}...")
    theta_0, model_0_sha256, model_0_hash, model_0_meta = load_model(model_0_path, device)
    if mode != "NoIn":
        interp_method = 0
        model_1_path = os.path.join(args.model_path, args.model_1)
        model_1_name = args.m1_name if args.m1_name is not None else os.path.splitext(os.path.basename(model_1_path))[0]
        print(f"Loading {model_1_name}...")
        theta_1, model_1_sha256, model_1_hash, model_1_meta = load_model(model_1_path, device)
        weights_a, alpha, alpha_info = parse_ratio(args.alpha, alpha_info, deep_a)
        if mode in ["sAD", "AD", "TRS", "ST", "TD","SIM","MD"]:
            model_2_path = os.path.join(args.model_path, args.model_2)
            model_2_name = args.m2_name if args.m2_name is not None else os.path.splitext(os.path.basename(model_2_path))[0]
            print(f"Loading {model_2_name}...")
            theta_2, model_2_sha256, model_2_hash, model_2_meta = load_model(model_2_path, device)
        if mode in ["TRS","ST","TS","SIM","MD"]:
            usebeta = True
            weights_b, beta, beta_info = parse_ratio(args.beta, beta_info, deep_b)
else:
    print(sha256(model_0_path, f"checkpoint/{os.path.splitext(os.path.basename(model_0_path))[0]}"))
    meta = read_metadata_from_safetensors(model_0_path)
    print(json.dumps(meta, indent=2))
    with open("./" + output_name + ".json", mode="a+") as dmp:
        json.dump(meta, dmp, indent=4)
    exit()
	
if args.vae is not None:
    vae_name = os.path.splitext(os.path.basename(args.vae))[0]
    vae, _ , _ , _ = load_model(args.vae, device, sha256=False)

if mode == "DARE":
    rand_generator = torch.Generator()
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

def blocker(blocks,blockids):
    blocks = blocks.split(" ")
    output = ""
    for w in blocks:
        flagger=[False]*len(blockids)
        changer = True
        if "-" in w:
            wt = [wt.strip() for wt in w.split('-')]
            if  blockids.index(wt[1]) > blockids.index(wt[0]):
                flagger[blockids.index(wt[0]):blockids.index(wt[1])+1] = [changer]*(blockids.index(wt[1])-blockids.index(wt[0])+1)
            else:
                flagger[blockids.index(wt[1]):blockids.index(wt[0])+1] = [changer]*(blockids.index(wt[0])-blockids.index(wt[1])+1)
        else:
            output = output + " " + w if output else w
        for i in range(len(blockids)):
            if flagger[i]: output = output + " " + blockids[i] if output else blockids[i]
    return output

def blockfromkey(key,isxl):
    if not isxl:
        re_inp = re.compile(r'\.input_blocks\.(\d+)\.')  # 12
        re_mid = re.compile(r'\.middle_block\.(\d+)\.')  # 1
        re_out = re.compile(r'\.output_blocks\.(\d+)\.') # 12

        weight_index = -1

        NUM_INPUT_BLOCKS = 12
        NUM_MID_BLOCK = 1
        NUM_OUTPUT_BLOCKS = 12
        NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

        if 'time_embed' in key:
            weight_index = -2                # before input blocks
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
        return BLOCKID[weight_index+1] ,BLOCKID[weight_index+1] 

    else:
        if not ("weight" in key or "bias" in key):return "Not Merge","Not Merge"
        if "label_emb" in key or "time_embed" in key: return "Not Merge","Not Merge"
        if "conditioner.embedders" in key : return "BASE","BASE"
        if "first_stage_model" in key : return "VAE","BASE"
        if "model.diffusion_model" in key:
            if "model.diffusion_model.out." in key: return "OUT8","OUT08"
            block = re.findall(r'input|mid|output', key)
            block = block[0].upper().replace("PUT","") if block else ""
            nums = re.sub(r"\D", "", key)[:1 if "MID" in block else 2] + ("0" if "MID" in block else "")
            add = re.findall(r"transformer_blocks\.(\d+)\.",key)[0] if "transformer" in key else ""
            return block + nums + add, block + "0" + nums[0] if "MID" not in block else "M00"

    return "Not Merge", "Not Merge"

def elementals(key,weight_index,deep,current_alpha):
    skey = key + BLOCKID[weight_index+1]
    for d in deep:
        if d.count(":") != 2 :continue
        dbs,dws,dr = d.split(":")[0],d.split(":")[1],d.split(":")[2]
        dbs = blocker(dbs,BLOCKID)
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
    return current_alpha

theta_funcs = {
    "WS":   (filename_weighted_sum, None, weighted_sum,"Weighted Sum"),
    "AD":   (filename_add_difference, get_difference, add_difference, "Add Difference"),
    "sAD":  (filename_add_difference, get_difference, add_difference, "Smooth Add Difference"),
    "MD":   (filename_add_difference, None, multiply_difference, "Multiply Difference"),
    "SIM":  (filename_add_difference, None, similarity_add_difference, "Similarity Add Difference"),
    "TD":   (filename_add_difference, None, add_difference, "Training Difference"),
    "TS":   (filename_weighted_sum, None, weighted_sum, "Tensor Sum"),
    "TRS":  (filename_triple_sum, None, triple_sum, "Triple Sum"),
    "ST":   (filename_sum_twice, None, sum_twice, "Sum Twice"),
    "NoIn": (filename_nothing, None, None, "No Interpolation"),
    "SIG":  (filename_sigmoid, None, sigmoid, "Sigmoid"),
    "GEO":  (filename_geom, None, geom, "Geometric"),
    "MAX":  (filename_max, None, weight_max, "Max"),
    "DARE": (filename_weighted_sum, None, dare_merge, "DARE")
}
filename_generator, theta_func1, theta_func2, merge_name = theta_funcs[mode] 

if theta_func1:
  pr = tqdm(theta_1.keys(), desc="Getting Difference of Model 1 and 2")
  for key in pr:
    if 'model' in key:
      if key in theta_2:
          t2 = theta_2.get(key, torch.zeros_like(theta_1[key]))
          theta_1[key] = theta_func1(theta_1[key], t2)
      else:
          theta_1[key] = torch.zeros_like(theta_1[key])
  del theta_2

if mode == "TS":
    theta_t = theta_0
    theta_0 ={}
    for key in theta_t:
        theta_0[key] = theta_t[key].clone()
    del theta_t

if args.use_dif_21:
    theta_3 = copy.deepcopy(theta_1)
    func1 = get_difference
    pr = tqdm(theta_2.keys(), desc="Getting Difference of Model 1 and 2")
    for key in pr:
        if 'model' in key:
          if key in theta_3:
              t2 = theta_3.get(key, torch.zeros_like(theta_2[key]))
              theta_2[key] = func1(theta_2[key], t2)
          else:
              theta_2[key] = torch.zeros_like(theta_2[key])
    del theta_3

if args.use_dif_10:
    theta_3 = copy.deepcopy(theta_0)
    func1 = get_difference
    pr = tqdm(theta_1.keys(), desc="Getting Difference of Model 0 and 1")
    for key in pr:
        if 'model' in key:
          if key in theta_3:
              t2 = theta_3.get(key, torch.zeros_like(theta_1[key]))
              theta_1[key] = func1(theta_1[key], t2)
          else:
              theta_1[key] = torch.zeros_like(theta_1[key])
    del theta_3

if args.use_dif_20:
    theta_3 = copy.deepcopy(theta_0)
    func1 = get_difference
    pr = tqdm(theta_2.keys(), desc="Getting Difference of Model 0 and 2")
    for key in pr:
        if 'model' in key:
          if key in theta_3:
              t2 = theta_3.get(key, torch.zeros_like(theta_2[key]))
              theta_2[key] = func1(theta_2[key], t2)
          else:
              theta_2[key] = torch.zeros_like(theta_2[key])
    del theta_3

    
if cosine0: #favors modelA's structure with details from B
    sim = torch.nn.CosineSimilarity(dim=0)
    sims = np.array([], dtype=np.float64)
    pr = tqdm(theta_0.keys(), desc="Caluculating Cosine 0")
    for key in pr:
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
    pr = tqdm(theta_0.keys(), desc="Caluculating Cosine 1")
    for key in pr:
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
  isxl = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in theta_1.keys()
  if args.fine is not None:
    fine = [float(t) for t in args.fine.split(",")]
    fine = fineman(fine,isxl)
  else:
    fine = ""
  if isxl and useblocks:
    if len(weights_a) == 25:
        weights_a = weighttoxl(weights_a)
        print(f"alpha weight converted for XL{weights_a}")
    if usebeta:
        if len(weights_b) == 25:
            weights_b = weighttoxl(weights_b)
            print(f"beta weight converted for XL{weights_b}")
    if len(weights_a) == 19: weights_a = weights_a + [0]
    if usebeta and len(weights_b) == 19: weights_b = weights_b + [0]
  pr = tqdm(theta_0.keys(), desc=f"{merge_name} Merging...")
  for key in pr:
    if args.vae is None and "first_stage_model" in key: continue
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
      al = list(theta_0[key].shape)
      bl = list(theta_1[key].shape)
      if usebeta:
        c = theta_2[key]
        cl = list(theta_2[key].shape)
      # check weighted and U-Net or not
      
      block,blocks26 = blockfromkey(key,isxl)
      if block == "Not Merge": continue
      weight_index = BLOCKIDXLL.index(blocks26) if isxl else BLOCKID.index(blocks26)
      if weight_index > 0:
            if weights_a is not None:
              current_alpha = weights_a[weight_index-1]
            if usebeta and weights_b is not None:
              current_beta = weights_b[weight_index-1]
			
      if len(deep_a) > 0:
        current_alpha = elementals(key,weight_index,deep_a,current_alpha)

      if len(deep_b) > 0:
        current_beta = elementals(key,weight_index,deep_b,current_beta)
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
            k = (((combined_similarity - sims.min()) / (sims.max() - sims.min()))- abs(current_alpha)).clip(min=0,max=1.0)
            theta_0[key] = b * (1 - k) + a * k
	
      elif cosine1:
        # skip VAE model parameters to get better results
        if "first_stage_model" in key: continue
        if "model" in key and key in theta_0:
            simab = sim(a.to(torch.float32), b.to(torch.float32))
            dot_product = torch.dot(a.view(-1).to(torch.float32), b.view(-1).to(torch.float32))
            magnitude_similarity = dot_product / (torch.norm(a.to(torch.float32)) * torch.norm(b.to(torch.float32)))
            combined_similarity = (simab + magnitude_similarity) / 2.0
            k = (((combined_similarity - sims.min()) / (sims.max() - sims.min()))- abs(current_alpha)).clip(min=0,max=1.0)
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
        if al != bl and al[0:1] + al[2:] == bl[0:1] + bl[2:]:
          ad = a[:, 0:4, :, :]
          result_is_instruct_pix2pix_model = (al[1] == 8 and bl[1] == 4)
          assert al[1] == 9 and bl[1] == 4, f"Bad dimensions for merged layer {key}: A={al}, B={bl}"
          result_is_inpainting_model = not result_is_instruct_pix2pix_model
        else:
          ad = a
          result_is_instruct_pix2pix_model = False
          result_is_inpainting_model = False
        if usebeta:
          theta_0[key] = theta_func2(ad, b, c, current_alpha, current_beta)
        else:
          theta_0[key] = theta_func2(ad, b, current_alpha)
      if any(item in key for item in FINETUNES) and fine:
            index = FINETUNES.index(key)
            if 5 > index : 
                theta_0[key] =theta_0[key]* fine[index] 
            else :theta_0[key] =theta_0[key] + torch.tensor(fine[5]).to(theta_0[key].device)
  pr = tqdm(theta_1.keys(), desc="Remerging...")
  for key in pr:
        if key in checkpoint_dict_skip_on_merge:
            continue
        if "model" in key and key not in theta_0:
            try:
                if mode in ["TRS","ST"] and theta_2:
                    b = theta_1[key]
                    if key in theta_2:
                        c = theta_2[key]
                        current_beta = beta
                        
                        block,blocks26 = blockfromkey(key,isxl)
                        if block == "Not Merge": continue
                        weight_index = BLOCKIDXLL.index(blocks26) if isxl else BLOCKID.index(blocks26)
                        if weight_index >= 0:
                            if weights_b is not None:
                                current_beta = weights_b[weight_index]
                        if len(deep_b) > 0:
                            current_beta = elementals(key,weight_index,deep_b,current_beta)
                        theta_0[key] = weighted_sum(b, c, current_beta)
                    else:
                        theta_0.update({key:theta_1[key]})
            except NameError:
                theta_0.update({key:theta_1[key]})
  del theta_1
  try:
    if theta_2:
        pr = tqdm(theta_2.keys(), desc="Remerging...")
        for key in pr:
            if key in checkpoint_dict_skip_on_merge:
                continue
            if "model" in key and key not in theta_0:
                theta_0.update({key:theta_2[key]})
        del theta_2
  except NameError:
    pass

else:
    isxl = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in theta_0.keys()
    if args.fine is not None:
        fine = [float(t) for t in args.fine.split(",")]
        fine = fineman(fine,isxl)
        pr = tqdm(theta_0.keys(), desc="Fine Tuning ...")
        for key in pr:
            if args.vae is None and "first_stage_model" in key: continue
            if any(item in key for item in FINETUNES) and fine:
                index = FINETUNES.index(key)
                if 5 > index : 
                    theta_0[key] =theta_0[key]* fine[index] 
                else :theta_0[key] =theta_0[key] + torch.tensor(fine[5]).to(theta_0[key].device)
    else:
        fine = ""
if args.vae is not None:
    pr = tqdm(vae.keys(), desc=f"Baking in VAE[{vae_name}] ...")
    for key in pr:
        theta_0_key = 'first_stage_model.' + key
        if theta_0_key in theta_0:
            theta_0[theta_0_key] = to_half(vae[key], args.save_half)
    del vae
isxl = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in theta_0
if isxl:
    # prune share memory tensors, "cond_stage_model." prefixed base tensors are share memory with "conditioner." prefixed tensors
    for i, key in enumerate(theta_0.keys()):
        if "cond_stage_model." in key:
            del theta_0[key]

theta_0 = to_half_k(theta_0, args.save_half)
if args.prune:
    theta_0 = prune_model(theta_0, "Model", isxl)
# for safetensors contiguous error
pr = tqdm(theta_0.keys(), desc="Check contiguous...")
for key in pr:
    v = theta_0[key]
    v = v.contiguous()
    theta_0[key] = v 

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

if args.save_quarter:
    fp = "fp8"
elif args.save_half:
    fp = "fp16"
else:
    fp = "fp32"
merge_recipe = {
"type": "merge-models-chattiori", # indicate this model was merged with chattiori's model mereger
"primary_model_hash": model_0_sha256,
"secondary_model_hash": model_1_sha256 if mode != "NoIn" else None,
"tertiary_model_hash": model_2_sha256 if mode in ["sAD", "AD", "TRS", "ST","TD","SIM","MD"] else None,
"merge_method": real_mode[mode],
"block_weights": (weights_a is not None or weights_b is not None),
"alpha_info": alpha_info if alpha_info != "" else None,
"beta_info": beta_info if beta_info != "" else None,
"calculation": calcl,
"fp": fp,
"output_name": output_name,
"bake_in_vae": vae_name if args.vae is not None else False,
"pruned": args.prune
}
metadata["sd_merge_recipe"] = json.dumps(merge_recipe)

def add_model_metadata(s256, hashed, meta, model_name):
  metadata["sd_merge_models"][s256] = {
  "name": model_name,
  "legacy_hash": hashed,
  "sd_merge_recipe": meta.get("sd_merge_recipe", None)
  }

  metadata["sd_merge_models"].update(meta.get("sd_merge_models", {}))

add_model_metadata(model_0_sha256, model_0_hash, model_0_meta, model_0_name)
if mode != "NoIn":
  add_model_metadata(model_1_sha256, model_1_hash, model_1_meta, model_1_name)
if mode in ["sAD", "AD", "TRS", "ST","TD","SIM","MD"]:
  add_model_metadata(model_2_sha256, model_2_hash, model_2_meta, model_2_name)

metadata["sd_merge_models"] = json.dumps(metadata["sd_merge_models"])

# check if output file already exists, ask to overwrite
print(f"Saving as {output_file}...")
if args.delete_source:
    os.remove(model_0_path)
    if mode != "NoIn":
      os.remove(model_1_path)
    if mode in ["sAD", "AD", "TRS", "ST","TD","SIM","MD"]:
      os.remove(model_2_path)
if args.save_safetensors:
    with torch.no_grad():
      if args.no_metadata:
        safetensors.torch.save_file(theta_0, output_path)
      else:
        safetensors.torch.save_file(theta_0, output_path, metadata=metadata)
else:
    torch.save({"state_dict": theta_0}, output_path)
del theta_0
file_size = round(os.path.getsize(output_path) / 1073741824,2)
print(f"Done! ({file_size}G)")
