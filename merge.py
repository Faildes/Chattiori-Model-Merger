import os
import argparse
import torch
import safetensors.torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Merge two models")
parser.add_argument("mode", type=str, help="Merging mode")
parser.add_argument("model_0", type=str, help="Path to model 0")
parser.add_argument("model_1", type=str, help="Path to model 1")
#parser.add_argument("--model_2", type=str, help="Optional, Path to model 2", default=None, required=False)
parser.add_argument("--vae", type=str, help="Path to vae", default=None, required=False)
parser.add_argument("--alpha", type=float, help="Alpha value, optional, defaults to 0.5", default=0.5, required=False)
parser.add_argument("--save_half", action="store_true", help="Save as float16", default=False, required=False)
parser.add_argument("--save_safetensors", action="store_true", help="Save as .safetensors", default=False, required=False)
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)

def to_half(tensor, enable):
    if enable and tensor.dtype == torch.float:
        return tensor.half()

    return tensor

def weighted_sum(theta0, theta1, alpha):
    return ((1 - alpha) * theta0) + (alpha * theta1)

def get_difference(theta1, theta2):
    return theta1 - theta2

def add_difference(theta0, theta1_2_diff, alpha):
    return theta0 + (alpha * theta1_2_diff)

args = parser.parse_args()
device = args.device

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
        device = map_location or shared.weight_load_location or devices.get_optimal_device_name()
        pl_sd = safetensors.torch.load_file(checkpoint_file, device=device)
    else:
        pl_sd = torch.load(checkpoint_file, map_location=map_location or shared.weight_load_location)

    if print_global_state and "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    sd = get_state_dict_from_checkpoint(pl_sd)
    return sd

if mode == "WS":
    _, extension_0 = os.path.splitext(args.model_0)
    if extension_0.lower() == ".safetensors":
        model_0 = safetensors.torch.load_file(args.model_0, device=device)
    else:
        model_0 = torch.load(args.model_0, map_location=device)
    _, extension_1 = os.path.splitext(args.model_1)
    if extension_1.lower() == ".safetensors":
        model_1 = safetensors.torch.load_file(args.model_1, device=device)
    else:
        model_1 = torch.load(args.model_1, map_location=device)
    if args.vae is not None:
        _, extension_vae = os.path.splitext(args.vae)
        if extension_vae.lower() == ".safetensors":
            vae = safetensors.torch.load_file(args.vae, device=device)
        else:
            vae = torch.load(args.vae, map_location=device)
    theta_0 = read_state_dict(model_0, map_location=device)
    theta_1 = read_state_dict(model_1, map_location=device)
#elif mode == "AD":
#    model_0 = torch.load(args.model_0, map_location=device)
#    model_1 = torch.load(args.model_1, map_location=device)
#    model_2 = torch.load(args.model_2, map_location=device)
#    if args.vae is not None:
#        vae = torch.load(args.vae, map_location=device)
#    theta_0 = model_0["state_dict"]
#    theta_1 = model_1["state_dict"]
#    theta_2 = model_2["state_dict"]
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

if discard_weights:
    regex = re.compile(discard_weights)
    for key in list(theta_0):
        if re.search(regex, key):
            theta_0.pop(key, None)
                
for key in tqdm(theta_0.keys(), desc="Stage 1/2"):
    if "model" in key and key in theta_1:
        theta_0[key] = (1 - alpha) * theta_0[key] + alpha * theta_1[key]

for key in tqdm(theta_1.keys(), desc="Stage 2/2"):
    if "model" in key and key not in theta_0:
        theta_0[key] = theta_1[key]

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
            theta_0[theta_0_key] = to_half(vae_dict[key], save_as_half)

if args.save_half:
    for key in theta_0.keys():
        theta_0[key] = to_half(theta_0[key], args.save_half)   
    
print("Saving...")
if args.save_safetensors:
    safetensors.torch.save_file({"state_dict": theta_0}, output_file, metadata={"format": "pt"})
else:
    torch.save({"state_dict": theta_0}, output_file)

print("Done!")
