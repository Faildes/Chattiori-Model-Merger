import os
import copy
import argparse
import torch
import re
import shutil
import safetensors.torch
from tqdm import tqdm

EMA_PREFIX = "model_ema."

METADATA = {'epoch': 0, 'global_step': 0, 'pytorch-lightning_version': '1.6.0'}

IDENTIFICATION = {
    "VAE": {
        "SD-v1": 0,
        "SD-v2": 869,
        "NAI": 2982,
        "WD-VAE-v1": 155,
        "WD-VAE-v2": 41
    },
    "CLIP-v1": {
        "SD-v1": 0,
    },
    "CLIP-v2": {
        "SD-v2": 1141,
        "WD-v1-4": 2543
    }
}

COMPONENTS = {
    "UNET-v1-SD": {
        "keys": {},
        "source": "UNET-v1-SD.txt",
        "prefix": "model.diffusion_model."
    },
    "UNET-v1-EMA": {
        "keys": {},
        "source": "UNET-v1-EMA.txt",
        "prefix": "model_ema.diffusion_model"
    },
    "UNET-v1-Inpainting": {
        "keys": {},
        "source": "UNET-v1-Inpainting.txt",
        "prefix": "model.diffusion_model."
    },
    "UNET-v1-Pix2Pix": {
        "keys": {},
        "source": "UNET-v1-Pix2Pix.txt",
        "prefix": "model.diffusion_model."
    },
    "UNET-v1-Pix2Pix-EMA": {
        "keys": {},
        "source": "UNET-v1-Pix2Pix-EMA.txt",
        "prefix": "model_ema.diffusion_model"
    },
    "UNET-v2-SD": {
        "keys": {},
        "source": "UNET-v2-SD.txt",
        "prefix": "model.diffusion_model."
    },
    "UNET-v2-Depth": {
        "keys": {},
        "source": "UNET-v2-Depth.txt",
        "prefix": "model.diffusion_model."
    },
    "VAE-v1-SD": {
        "keys": {},
        "source": "VAE-v1-SD.txt",
        "prefix": "first_stage_model."
    },
    "CLIP-v1-SD": {
        "keys": {},
        "source": "CLIP-v1-SD.txt",
        "prefix": "cond_stage_model.transformer.text_model."
    },
    "CLIP-v1-NAI": {
        "keys": {},
        "source": "CLIP-v1-SD.txt",
        "prefix": "cond_stage_model.transformer."
    },
    "CLIP-v2-SD": {
        "keys": {},
        "source": "CLIP-v2-SD.txt",
        "prefix": "cond_stage_model.model."
    },
    "CLIP-v2-WD": {
        "keys": {},
        "source": "CLIP-v2-WD.txt",
        "prefix": "cond_stage_model.model."
    },
    "Depth-v2-SD": {
        "keys": {},
        "source": "Depth-v2-SD.txt",
        "prefix": "depth_model.model."
    },
    "LoRA-v1-CLIP": {
        "keys": {},
        "shapes": {},
        "source": "LoRA-v1-CLIP.txt",
        "prefix": ""
    },
    "LoRA-v1A-CLIP": {
        "keys": {},
        "shapes": {},
        "source": "LoRA-v1A-CLIP.txt",
        "prefix": ""
    },
    "LoRA-v1-UNET": {
        "keys": {},
        "shapes": {},
        "source": "LoRA-v1-UNET.txt",
        "prefix": ""
    },
    "LoRA-v1A-UNET": {
        "keys": {},
        "shapes": {},
        "source": "LoRA-v1A-UNET.txt",
        "prefix": ""
    },
    "ControlNet-v1-SD": {
        "keys": {},
        "shapes": {},
        "source": "ControlNet-v1-SD.txt",
        "prefix": "control_model."
    },
}

COMPONENT_CLASS = {
    "UNET-v1-SD": "UNET-v1",
    "UNET-v1-EMA": "EMA-UNET-v1",
    "UNET-v1-Inpainting": "UNET-v1",
    "UNET-v1-Pix2Pix": "UNET-v1-Pix2Pix",
    "UNET-v1-Pix2Pix-EMA": "EMA-UNET-v1-Pix2Pix",
    "UNET-v2-SD": "UNET-v2",
    "UNET-v2-Depth": "UNET-v2-Depth",
    "VAE-v1-SD": "VAE-v1",
    "CLIP-v1-SD": "CLIP-v1",
    "CLIP-v1-NAI": "CLIP-v1",
    "CLIP-v2-SD": "CLIP-v2",
    "CLIP-v2-WD": "CLIP-v2",
    "Depth-v2-SD": "Depth-v2",
    "LoRA-v1-UNET": "LoRA-v1-UNET",
    "LoRA-v1-CLIP": "LoRA-v1-CLIP",
    "LoRA-v1A-UNET": "LoRA-v1-UNET",
    "LoRA-v1A-CLIP": "LoRA-v1-CLIP",
    "ControlNet-v1-SD": "ControlNet-v1",
}

OPTIONAL = [
    ("alphas_cumprod", (1000,)),
    ("alphas_cumprod_prev", (1000,)),
    ("betas", (1000,)),
    ("log_one_minus_alphas_cumprod", (1000,)),
    ("model_ema.decay", ()),
    ("model_ema.num_updates", ()),
    ("posterior_log_variance_clipped", (1000,)),
    ("posterior_mean_coef1", (1000,)),
    ("posterior_mean_coef2", (1000,)),
    ("posterior_variance", (1000,)),
    ("sqrt_alphas_cumprod", (1000,)),
    ("sqrt_one_minus_alphas_cumprod", (1000,)),
    ("sqrt_recip_alphas_cumprod", (1000,)),
    ("sqrt_recipm1_alphas_cumprod", (1000,)),
    ("logvar", (1000,)),
]

ARCHITECTURES = {
    "UNET-v1": {
        "classes": ["UNET-v1"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "UNET-v1-Pix2Pix": {
        "classes": ["UNET-v1-Pix2Pix"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "UNET-v2": {
        "classes": ["UNET-v2"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "UNET-v2-Depth": {
        "classes": ["UNET-v2-Depth"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "VAE-v1": {
        "classes": ["VAE-v1"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "CLIP-v1": {
        "classes": ["CLIP-v1"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "CLIP-v2": {
        "classes": ["CLIP-v2"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "Depth-v2": {
        "classes": ["Depth-v2"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "ControlNet-v1": {
        "classes": ["ControlNet-v1"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "SD-v1": {
        "classes": ["UNET-v1", "VAE-v1", "CLIP-v1"],
        "optional": OPTIONAL,
        "required": [],
        "prefixed": True
    },
    "SD-v1-Pix2Pix": {
        "classes": ["UNET-v1-Pix2Pix", "VAE-v1", "CLIP-v1"],
        "optional": OPTIONAL,
        "required": [],
        "prefixed": True
    },
    "SD-v1-ControlNet": {
        "classes": ["UNET-v1", "VAE-v1", "CLIP-v1", "ControlNet-v1"],
        "optional": OPTIONAL,
        "required": [],
        "prefixed": True
    },
    "SD-v2": {
        "classes": ["UNET-v2", "VAE-v1", "CLIP-v2"],
        "optional": OPTIONAL,
        "required": [],
        "prefixed": True
    },
    "SD-v2-Depth": {
        "classes": ["UNET-v2-Depth", "VAE-v1", "CLIP-v2", "Depth-v2"],
        "optional": OPTIONAL,
        "required": [],
        "prefixed": True
    },
    "EMA-v1": {
        "classes": ["EMA-UNET-v1"],
        "optional": OPTIONAL,
        "required": [],
        "prefixed": True
    },
    "EMA-v1-Pix2Pix": {
        "classes": ["EMA-UNET-v1-Pix2Pix"],
        "optional": OPTIONAL,
        "required": [],
        "prefixed": True
    },
    # standalone component architectures, for detecting broken models
    "UNET-v1-BROKEN": {
        "classes": ["UNET-v1"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "UNET-v1-Pix2Pix-BROKEN": {
        "classes": ["UNET-v1-Pix2Pix"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "UNET-v2-BROKEN": {
        "classes": ["UNET-v2"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "UNET-v2-Depth-BROKEN": {
        "classes": ["UNET-v2-Depth"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "VAE-v1-BROKEN": {
        "classes": ["VAE-v1"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "CLIP-v1-BROKEN": {
        "classes": ["CLIP-v1"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "CLIP-v2-BROKEN": {
        "classes": ["CLIP-v2"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "Depth-v2-BROKEN": {
        "classes": ["Depth-v2"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "ControlNet-v1-BROKEN": {
        "classes": ["ControlNet-v1"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "LoRA-v1-UNET": {
        "classes": ["LoRA-v1-UNET"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "LoRA-v1-CLIP": {
        "classes": ["LoRA-v1-CLIP"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "LoRA-v1": {
        "classes": ["LoRA-v1-CLIP", "LoRA-v1-UNET"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
}

parser = argparse.ArgumentParser(description="Merge two models")
parser.add_argument("mode", type=str, help="Merging mode")
parser.add_argument("model_path", type=str, help="Path to models")
parser.add_argument("model_0", type=str, help="Name of model 0")
parser.add_argument("model_1", type=str, help="Name of model 1")
parser.add_argument("--model_2", type=str, help="Optional, Name of model 2", default=None, required=False)
parser.add_argument("--vae", type=str, help="Path to vae", default=None, required=False)
parser.add_argument("--alpha", type=float, help="Alpha value, optional, defaults to 0.5", default=0.5, required=False)
parser.add_argument("--save_half", type=bool, help="Save as float16", default=False, required=False)
parser.add_argument("--prune", type=bool, help="Prune Model", default=False, required=False)
parser.add_argument("--save_safetensors", type=bool, help="Save as .safetensors", default=False, required=False)
parser.add_argument("--keep_ema", type=bool, help="Keep ema", default=False, required=False)
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)

def to_half(tensor, enable):
    if enable and tensor.dtype == torch.float:
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
        
def weighted_sum(theta0, theta1, alpha):
    return ((1 - alpha) * theta0) + (alpha * theta1)

def get_difference(theta1, theta2):
    return theta1 - theta2

def add_difference(theta0, theta1_2_diff, alpha):
    return theta0 + (alpha * theta1_2_diff)

args = parser.parse_args()
device = args.device
mode = args.mode

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
model_1_path = os.path.join(args.model_path, args.model_1)

if args.model_2 is not None:
  model_2_path = os.path.join(args.model_path, args.model_2)
if mode == "WS":
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
alpha = args.alpha

if args.save_safetensors:
    output_file = f'{args.output}.safetensors'
else:
    output_file = f'{args.output}.ckpt'

# check if output file already exists, ask to overwrite
if os.path.isfile(output_file):
    print("Output file already exists. Overwrite? (y/n)")
    while True:
        overwrite = input()
        if overwrite == "y":
            break
        elif overwrite == "n":
            print("Exiting...")
            exit()
        else:
            print("Please enter y or n")

model_0_name = os.path.splitext(os.path.basename(model_0_path))[0]
model_1_name = os.path.splitext(os.path.basename(model_1_path))[0]
if args.model_2 is not None:
  model_2_name = os.path.splitext(os.path.basename(model_2_path))[0]

def filename_weighted_sum():
  a = model_0_name
  b = model_1_name
  Ma = round(1 - alpha, 2)
  Mb = round(alpha, 2)

  return f"{Ma}({a}) + {Mb}({b})"

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
              theta_0[key][:, 0:4, :, :] = theta_func2(a[:, 0:4, :, :], b, alpha)#Merge only the vectors the models have in common.  Otherwise we get an error due to dimension mismatch.
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
    
if args.save_half and mode != "NoIn":
    for key in theta_0.keys():
        theta_0[key] = to_half(theta_0[key], args.save_half)   

def get_prefixed_keys(component):
  prefix = COMPONENTS[component]["prefix"]
  allowed = COMPONENTS[component]["keys"]
  return set([(prefix + k, z) for k, z in allowed])

def get_allowed_keys(arch, allowed_classes=None):
  # get all allowed keys
  allowed = set()
  for a in arch:
      if allowed_classes == None:
          allowed.update(ARCHITECTURES[a]["required"])
          allowed.update(ARCHITECTURES[a]["optional"])
      prefixed = ARCHITECTURES[a]["prefixed"]
      for clss in arch[a]:
          if allowed_classes == None or clss in allowed_classes:
              for comp in arch[a][clss]:
                  comp_keys = COMPONENTS[comp]["keys"]
                  if prefixed:
                      comp_keys = get_prefixed_keys(comp)
                  allowed.update(comp_keys)
  return allowed

def tensor_shape(key, data):
  if hasattr(data, 'shape'):
      shape = tuple(data.shape)
      for c in ["LoRA-v1-UNET", "LoRA-v1-CLIP"]:
          if key in COMPONENTS[c]['shapes']:
              lora_shape = COMPONENTS[c]['shapes'][key]
              if len(shape) == len(lora_shape):
                  shape = tuple(a if b != -1 else b for a, b in zip(shape, lora_shape))
      return shape
  return tuple()

def prune_model(model, arch, keep_ema, dont_half):
  allowed = get_allowed_keys(arch)
  for k in list(model.keys()):
      kk = (k, tensor_shape(k, model[k]))
      keep = False
      if kk in allowed:
          keep = True
      if k.startswith(EMA_PREFIX) and keep_ema:
          keep = True
      if not keep:
          del model[k]
          continue
      if not dont_half and type(model[k]) == torch.Tensor and model[k].dtype == torch.float32:
          model[k] = model[k].half()

loaded = None
model_path = args.model_path
output_path = os.path.join(model_path, output_file)
if args.prune:
  output_a = os.path.join(model_path, "test.ckpt")
  torch.save({"state_dict": theta_0}, output_a)
  print("Pruning...\n")
  sd = torch.load(output_a, map_location=device)
  nsd = dict()
  print(sd.keys())
  for k in sd.keys():
      if k != "optimizer_states":
          nsd[k] = sd[k]
  if "global_step" in sd:
      print(f"This is global step {sd['global_step']}.")
  if args.keep_ema:
      sd = nsd["state_dict"].copy()
      # infer ema keys
      ema_keys = {k: "model_ema." + k[6:].replace(".", ".") for k in sd.keys() if k.startswith("model.")}
      new_sd = dict()

      for k in sd:
          if k in ema_keys:
              new_sd[k] = sd[ema_keys[k]].half()
          elif not k.startswith("model_ema.") or k in ["model_ema.num_updates", "model_ema.decay"]:
              new_sd[k] = sd[k].half()

      assert len(new_sd) == len(sd) - len(ema_keys)
      nsd["state_dict"] = new_sd
  else:
      sd = nsd['state_dict'].copy()
      new_sd = dict()
      for k in sd:
          new_sd[k] = sd[k].half()
      nsd['state_dict'] = new_sd
  nsd = get_state_dict_from_checkpoint(nsd)
  print("Saving...")
  if args.save_safetensors:
    with torch.no_grad():
        safetensors.torch.save_file(nsd, output_path, metadata={"format": "pt"})
  else:
      torch.save({"state_dict": nsd}, output_path)
else:
  print("Saving...")
  if args.save_safetensors:
    with torch.no_grad():
        safetensors.torch.save_file(theta_0, output_path, metadata={"format": "pt"})
  else:
      torch.save({"state_dict": theta_0}, output_path)
output_path = os.path.join(model_path, output_file)

del theta_0
print("Done!")
