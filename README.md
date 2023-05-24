# Chattiori's Model Merger

This script merges stable-diffusion models with the settings of checkpoint merger at a user-defined ratio.  
You can use ckpt and safetensors as checkpoint file.  
This script isn't need to be loaded on GPU.

The mode is:

- "WS" for Weighted Sum
- "SIG" for Sigmoid Merge
- "GEO" for Geometric Merge
- "MAX" for Max Merge
- "AD" for Add Difference (requires model2)
- "sAD" for Smooth Add Difference (requires model2)
- "TRS" for Triple Sum (requires model2)
- "ST" for Sum Twice (requires model2)
- "NoIn" for No Interporation
- "RM" for Read Metadata

The ratio works as follows:

- 0.5 is a 50/50 mix of model0 and model1
- 0.3 is a 70/30 mix with more influence from model0 than model1

## Running it

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
  - Form: `python merge.py mode model_path model_0 model_1 --alpha 0.5 --output merged`
  - Example: `python merge.py "WS" "C:...\Model parent file path" "FILE A.ckpt" "FILE B.safetensors" --alpha 0.45 --vae "C:...\VAE.safetensors" --prune --save_half --output "MERGED"`
    - Optional: `--model_2` sets the tertiory model, if omitted
    - Optional: `--alpha` controls how much weight is put on the second model. Defaults to 0, if omitted  
    Can be written in float value, [Merge Block Weight type writing](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui/blob/master/README.md) and [Elemental Merge type writing](https://github.com/hako-mikan/sd-webui-supermerger/blob/main/elemental_en.md).
    - Optional: `--rand_alpha` randomizes weight put on the second model, if omitted  
    Need to be written in str like `"MIN, MAX, SEED"`.
    - Optional: `--beta` controls how much weight is put on the third model. Defaults to 0, if omitted  
    Can be written in float value, Merge Block Weight type writing and Elemental Merge type writing.
    - Optional: `--rand_beta` randomizes weight put on the third model, if omitted  
    Need to be written in str like `"MIN, MAX, SEED"`. 
    - Optional: `--vae` sets the vae file by set the path, if omitted
    - Optional: `--cosine0` determines to favor model0's structure with details from 1, if omitted
    - Optional: `--cosine1` determines to favor model1's structure with details from 0, if omitted
    - Optional: `--save_half` determines whether save the file as fp16, if omitted
    - Optional: `--prune` determines whether prune the model, if omitted
    - Optional: `--keep_ema` determines keep only ema while prune, if omitted
    - Optional: `--save_safetensors` determines whether save the file as safetensors, if omitted
    - Optional: `--output` is the filename of the merged file, without file extension. Defaults to "merged", if omitted
    - Optional: `--functn` determines whether add merge function names, if omitted
    - Optional: `--delete_source` determines whether to delete the source checkpoint files, if omitted
    - Optional: `--device` is the device that's going to be used to merge the models. Unless you have a ton of VRAM, you should probably just ignore this. Defaults to 'cpu', if omitted.
      - Required VRAM seems to be roughly equivalent to the size of `(size of both models) * 1.15`. Merging 2 models at 3.76GB resulted in rougly 8.6GB of VRAM usage on top of everything else going on.
      - If you have enough VRAM to merge on your GPU you can use `--device "cuda:x"` where x is the card corresponding to the output of `nvidia-smi -L`

### For Colab users

Install torch safetensors.  
`!pip install torch safetensors`

Clone this repo.  
`!cd /content/`  
`!git clone https://github.com/Faildes/merge-models`

Run merge.py.  
`!cd /content/merge-models/`  
`!python merge.py...`

## Potential Problems & Troubleshooting

- Depending on your operating system and specific installation of python you might need to replace `py` with `python`, `python3`, `conda` or something else entirely.

## Credits

- [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for overall designing.
- [eyriewow](https://github.com/eyriewow/merge-models) for original merge-models.
- [lopho](https://github.com/lopho/stable-diffusion-prune) for Pruning system.
- [idelairre](https://github.com/idelairre/sd-merge-models) for Geometric, Sigmoid and Max Sum.
- [hako-mikan](https://github.com/hako-mikan/sd-webui-supermerger) for Triple Sum, Sum twice, Smooth Add Difference, Cosine Merging and Elemental Merge.
- [bbc-mc](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui) for Block Weighted Merge.
- Eyriewow got the merging logic in `merge.py` from [this post](https://discord.com/channels/1010980909568245801/1011008178957320282/1018117933894996038) by r_Sh4d0w, who seems to have gotten it from [mlfoundations/wise-ft](https://github.com/mlfoundations/wise-ft)
