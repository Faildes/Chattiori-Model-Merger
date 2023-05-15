import glob
import os
import sys
import time

import copy
import argparse
import torch
import re
import shutil
import safetensors.torch
import safetensors
from tqdm import tqdm
            
parser = argparse.ArgumentParser(description="Merge two models")
parser.add_argument("mode", type=str, help="Merging mode")
parser.add_argument("model_path", type=str, help="Path to models")
parser.add_argument("model_0", type=str, help="Name of model 0")
parser.add_argument("model_1", type=str, help="Optional, Name of model 1", default=None)
parser.add_argument("--model_2", type=str, help="Optional, Name of model 2", default=None, required=False)
parser.add_argument("--vae", type=str, help="Path to vae", default=None, required=False)
parser.add_argument("--alpha", type=float, help="Alpha value, optional, defaults to 0.5", default=0.5, required=False)
parser.add_argument("--save_half", action="store_true", help="Save as float16", required=False)
parser.add_argument("--prune", action="store_true", help="Prune Model", required=False)
parser.add_argument("--save_safetensors", action="store_true", help="Save as .safetensors", required=False)
parser.add_argument("--keep_ema", action="store_true", help="Keep ema", required=False)
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
parser.add_argument("--functn", action="store_true", help="Add function name to the file", required=False)
parser.add_argument("--delete_source", action="store_true", help="Delete the source checkpoint file", required=False)
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)

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

def weight_max(theta0, theta1, alpha):
    return torch.max(theta0, theta1)

def geom(theta0, theta1, alpha):
    return torch.pow(theta0, 1 - alpha) * torch.pow(theta1, alpha)

def sigmoid(theta0, theta1, alpha):
    return (1 / (1 + torch.exp(-4 * alpha))) * (theta0 + theta1) - (1 / (1 + torch.exp(-alpha))) * theta0

def weighted_sum(theta0, theta1, alpha):
    return ((1 - alpha) * theta0) + (alpha * theta1)

def get_difference(theta1, theta2):
    return theta1 - theta2

def add_difference(theta0, theta1_2_diff, alpha):
    return theta0 + (alpha * theta1_2_diff)

args = parser.parse_args()
device = args.device
mode = args.mode

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
while os.path.isfile(output_path):
    print(f"{output_file} already exists. Overwrite? (y/n)")
    overwrite = input()
    if overwrite == "y":
        os.remove(output_path)
        break
    elif overwrite == "n":
        if args.save_safetensors:
            output_file = f"{output_name}_{fan}.safetensors"
        else:
            output_file = f"{output_name}_{fan}.ckpt"
        output_path = os.path.join(model_path, output_file)
        fan += 1
    else:
        print("")
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

model_0_path = os.path.join(args.model_path, args.model_0)
if args.model_1 is not None:
    model_1_path = os.path.join(args.model_path, args.model_1)
if args.model_2 is not None:
  model_2_path = os.path.join(args.model_path, args.model_2)

if mode in ["WS", "SIG", "GEO", "MAX"]:
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
  if args.vae is not None:
      _, extension_vae = os.path.splitext(args.vae)
      if extension_vae.lower() == ".safetensors":
          vae = safetensors.torch.load_file(args.vae, device=device)
      else:
          vae = torch.load(args.vae, map_location=device)
elif mode == "AD":
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
  if args.vae is not None:
      _, extension_vae = os.path.splitext(args.vae)
      if extension_vae.lower() == ".safetensors":
          vae = safetensors.torch.load_file(args.vae, device=device)
      else:
          vae = torch.load(args.vae, map_location=device)

elif mode == "NoIn":
  interp_method = 2
  _, extension_0 = os.path.splitext(model_0_path)
  if extension_0.lower() == ".safetensors":
      model_0 = safetensors.torch.load_file(model_0_path, device=device)
  else:
      model_0 = torch.load(model_0_path, map_location=device)
  if args.vae is not None:
      _, extension_vae = os.path.splitext(args.vae)
      if extension_vae.lower() == ".safetensors":
          vae = safetensors.torch.load_file(args.vae, device=device)
      else:
          vae = torch.load(args.vae, map_location=device)
  if args.prune:
    print("Pruning...\n")
    model = prune_model(model_0, args.keep_ema, args.save_half)
    print("Saving...")
    if args.save_safetensors:
      with torch.no_grad():
          safetensors.torch.save_file(model, output_path, metadata={"format": "pt"})
    else:
        torch.save({"state_dict": model}, output_path)
    del model
    exit()


alpha = args.alpha

model_0_name = os.path.splitext(os.path.basename(model_0_path))[0]
if args.model_1 is not None:
  model_1_name = os.path.splitext(os.path.basename(model_1_path))[0]
if args.model_2 is not None:
  model_2_name = os.path.splitext(os.path.basename(model_2_path))[0]
if args.prune:
  model_0 = prune_model(model_0)
  if args.model_1 is not None:
    model_1 = prune_model(model_1)
  if args.model_2 is not None:
    model_1 = prune_model(model_1)

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
  
theta_funcs = {
    "WS": (filename_weighted_sum, None, weighted_sum),
    "AD": (filename_add_difference, get_difference, add_difference),
    "NoIn": (filename_nothing, None, None),
    "SIG": (filename_sigmoid, None, sigmoid),
    "GEO": (filename_geom, None, geom),
    "MAX": (filename_max, None, weight_max),
}
filename_generator, theta_func1, theta_func2 = theta_funcs[mode] 

if theta_func2:
  print(f"Loading {model_1_name}...")
  theta_1 = read_state_dict(model_1_path, map_location=device)
else:
  theta_1 = None
        
if theta_func1:
  print(f"Loading {model_2_name}...")
  theta_2 = read_state_dict(model_2_path, map_location=device)
  for key in tqdm(theta_1.keys()):
    if key in checkpoint_dict_skip_on_merge:
      continue
    if 'model' in key:
      if key in theta_2:
          t2 = theta_2.get(key, torch.zeros_like(theta_1[key]))
          theta_1[key] = theta_func1(theta_1[key], t2)
      else:
          theta_1[key] = torch.zeros_like(theta_1[key])
  del theta_2

print(f"Loading {model_0_name}...")
theta_0 = read_state_dict(model_0_path, map_location=device)

for key in tqdm(theta_0.keys(), desc="Merging"):
    if theta_1 and "model" in key and key in theta_1:
      if key in checkpoint_dict_skip_on_merge:
        continue
      a = theta_0[key]
      b = theta_1[key]

      # this enables merging an inpainting model (A) with another one (B);
      # where normal model would have 4 channels, for latenst space, inpainting model would
      # have another 4 channels for unmasked picture's latent space, plus one channel for mask, for a total of 9
      if a.shape != b.shape and a.shape[0:1] + a.shape[2:] == b.shape[0:1] + b.shape[2:]:
          if a.shape[1] == 4 and b.shape[1] == 9:
              raise RuntimeError("When merging inpainting model with a normal one, A must be the inpainting model.")
          if a.shape[1] == 4 and b.shape[1] == 8:
              raise RuntimeError("When merging instruct-pix2pix model with a normal one, A must be the instruct-pix2pix model.")

          if a.shape[1] == 8 and b.shape[1] == 4:#If we have an Instruct-Pix2Pix model...
              theta_0[key][:, 0:4, :, :] = theta_func2(a[:, 0:4, :, :], b, alpha)
              result_is_instruct_pix2pix_model = True
          else:
              assert a.shape[1] == 9 and b.shape[1] == 4, f"Bad dimensions for merged layer {key}: A={a.shape}, B={b.shape}"
              theta_0[key][:, 0:4, :, :] = theta_func2(a[:, 0:4, :, :], b, alpha)
              result_is_inpainting_model = True
      else:
          theta_0[key] = theta_func2(a, b, alpha)
      
      theta_0[key] = to_half(theta_0[key], args.save_half)
del theta_1

vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}

def load_vae_dict(filename, map_location):
    vae_ckpt = read_state_dict(filename, map_location=map_location)
    vae_dict_1 = {k: v for k, v in vae_ckpt.items() if k[0:4] != "loss" and k not in vae_ignore_keys}
    return vae_dict_1
            
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
  print("Pruning...\n")
  output_a = os.path.join(model_path, "test.safetensors")
  if os.path.isfile(output_a):
    os.remove(output_a)
  safetensors.torch.save_file(theta_0, output_a,metadata={"format": "pt"})
  sd = safetensors.torch.load_file(output_a, device=device)
  model = prune_model(sd)
  print("Saving...")
  if args.save_safetensors:
    with torch.no_grad():
        safetensors.torch.save_file(model, output_path, metadata={"format": "pt"})
  else:
      torch.save({"state_dict": model}, output_path)
  del model
  os.remove(output_a)
else:
  print("Saving...")
  if args.save_safetensors:
    with torch.no_grad():
        safetensors.torch.save_file(theta_0, output_path, metadata={"format": "pt"})
  else:
      torch.save({"state_dict": theta_0}, output_path)
if args.delete_source:
    os.remove(model_0_path)
    if args.model_1 is not None:
      os.remove(model_1_path)
    if args.model_2 is not None:
      os.remove(model_2_path)
del theta_0
print("Done!")
