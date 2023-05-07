# Merge Models

This script merges stable-diffusion models with the settings of checkpoint merger at a user-defined ratio.

The mode is:

- "WS" for Weighted Sum
- "AD" for Add Difference (requires model2)
- "NoIn" for No Interporation

The ratio works as follows:

- 0.5 is a 50/50 mix of model0 and model1
- 0.3 is a 70/30 mix with more influence from model0 than model1

## Running it

### Bat file Method for Windows Users

If you are using Windows, the easiest way to use this script is to use the .bat file.

- Download this repo as a zip file
- Extract the folder and place it in the main folder of your stable-diffusion install
  - Copy the two models you want to merge into the folder you just created
  - Run `merge.bat`
  - The .bat file should guide you through the merge process

### Running merge.py Directly

If you aren't using Automatic's web UI or are comfortable with the command line, you can also run `merge.py` directly.
Just like with the .bat method, I'd recommend creating a folder within your stable-diffusion installation's main folder. This script requires torch to be installed, which you most likely will have installed in a venv inside your stable-diffusion webui install.
- Requires pytorch safetensors
- Navigate to the merge folder in your terminal
- Activate the venv
  - For users of Automatic's Webui use
    - `..\venv\Scripts\activate`
  - For users of [sd-webui](https://github.com/sd-webui/stable-diffusion-webui) (formerly known as HLKY) you should just be able to do
    - `conda activate ldm`
- run merge.py with arguments
  - `py merge.py mode model_path model_0 model_1 --alpha 0.5 --output merged`
    - Optional: `--model_2` sets the tertiory model
    - Optional: `--alpha` controls how much weight is put on the second model. Defaults to 0.5, if omitted
    - Optional: `--vae` sets the vae file by set the path
    - Optional: `--save_half` determines whether save the file as fp16 or not Defaults to False
    - Optional: `--prune` determines whether prune the model or not Defaults to False
    - Optional: `--keep_ema` determines keep only ema while prune Defaults to False
    - Optional: `--save_safetensors` determines whether save the file as safetensors Defaults to False
    - Optional: `--output` is the filename of the merged file, without file extension. Defaults to "merged", if omitted
    - Optional: `--device` is the device that's going to be used to merge the models. Unless you have a ton of VRAM, you should probably just ignore this. Defaults to 'cpu', if omitted.
      - Required VRAM seems to be roughly equivalent to the size of `(size of both models) * 1.15`. Merging 2 models at 3.76GB resulted in rougly 8.6GB of VRAM usage on top of everything else going on.
      - If you have enough VRAM to merge on your GPU you can use `--device "cuda:x"` where x is the card corresponding to the output of `nvidia-smi -L`

## Potential Problems & Troubleshooting

- Depending on your operating system and specific installation of python you might need to replace `py` with `python`, `python3`, `conda` or something else entirely.

## Credits

- Thanks to Automatic and his fantastic Webui, I stole some of the code for the `merge.bat` from him.
- I got the merging logic in `merge.py` from [this post](https://discord.com/channels/1010980909568245801/1011008178957320282/1018117933894996038) by r_Sh4d0w, who seems to have gotten it from [mlfoundations/wise-ft](https://github.com/mlfoundations/wise-ft)
