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
- "MD" for Multiply Difference (requires model2 and beta)
- "SIM" for Similarity Add Difference (requires model2 and beta)
- "TD" for Train Difference (requires model2)
- "TRS" for Triple Sum (requires model2 and beta)
- "TS" for Tensor Sum (requires beta)
- "ST" for Sum Twice (requires model2 and beta)
- "NoIn" for No Interporation
- "RM" for Read Metadata
- "DARE" for DARE Merge

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
    If SEED is not setted, it will be completely random (generates seed).  
    Or `"MIN, MAX, SEED, [Elemental merge args]"` if you want to specify.  
    Check out [Elemental Random](https://github.com/Faildes/merge-models/blob/main/elemental_random.md) for Elemental merge args.
    - Optional: `--beta` controls how much weight is put on the third model. Defaults to 0, if omitted  
    Can be written in float value, Merge Block Weight type writing and Elemental Merge type writing.
    - Optional: `--rand_beta` randomizes weight put on the third model, if omitted  
    Need to be written in str like `"MIN, MAX, SEED"`.   
    If SEED is not setted, it will be completely random (generates seed).  
    Or `"MIN, MAX, SEED, [Elemental merge args]"` if you want to specify.  
    Check out [Elemental Random](https://github.com/Faildes/merge-models/blob/main/elemental_random.md) for Elemental merge args.
    - Optional: `--vae` sets the vae file by set the path, if omitted.  
      If not, the vae stored inside the model will automatically discarded.
    - Optional: `--m0_name` determines the name that to write in the data for the model0, if omitted
    - Optional: `--m1_name` determines the name that to write in the data for the model1, if omitted
    - Optional: `--m2_name` determines the name that to write in the data for the model2, if omitted
    - Optional: `--cosine0` determines to favor model0's structure with details from 1, if omitted  
    Check out [Calcmode](https://github.com/hako-mikan/sd-webui-supermerger/blob/main/calcmode_en.md) by hako-mikan for the information.
    - Optional: `--cosine1` determines to favor model1's structure with details from 0, if omitted  
    Check out [Calcmode](https://github.com/hako-mikan/sd-webui-supermerger/blob/main/calcmode_en.md) by hako-mikan for the information.
    - Optional: `--use_dif_10` determines to use the difference between model0 and model1 as model1, if omitted
    - Optional: `--use_dif_20` determines to use the difference between model0 and model2 as model2, if omitted
    - Optional: `--use_dif_21` determines to use the difference between model2 and model1 as model2, if omitted
    - Optional: `--fine` determines adjustment of details, if omitted  
    Check out [Elemental EN](https://github.com/hako-mikan/sd-webui-supermerger/blob/main/elemental_en.md#adjust) by hako-mikan for the information.
    - Optional: `--save_half` determines whether save the file as fp16, if omitted
    - Optional: `--prune` determines whether prune the model, if omitted
    - Optional: `--keep_ema` determines keep only ema while prune, if omitted
    - Optional: `--save_safetensors` determines whether save the file as safetensors, if omitted
    - Optional: `--output` is the filename of the merged file, without file extension. Defaults to "merged", if omitted
    - Optional: `--functn` determines whether add merge function names, if omitted
    - Optional: `--delete_source` determines whether to delete the source checkpoint files, not vae file, if omitted
    - Optional: `--no_metadata` saves the checkpoint without metadata, if omitted
    - Optional: `--device` is the device that's going to be used to merge the models. Unless you have a ton of VRAM, you should probably just ignore this. Defaults to 'cpu', if omitted.
      - For `.ckpt` files, required VRAM seems to be roughly equivalent to the size of `(size of both models) * 1.15`. Merging 2 models at 3.76GB resulted in rougly 8.6GB of VRAM usage on top of everything else going on.
      - For `.safetensors`, required VRAM seems to be roughly equivalent to the size of `size of both models`. Merging 2 models at 3.76GB resulted in rougly 7.5GB of VRAM usage on top of everything else going on.
      - If you have enough VRAM to merge on your GPU you can use `--device "cuda:x"` where x is the card corresponding to the output of `nvidia-smi -L`

### For Colab users

- Install required scripts.  
`!pip install torch safetensors` *for safetensors*  
`!pip install pytorch_lighting` *for ckpt*  

- Clone this repo.  
`!cd /content/`  
`!git clone https://github.com/Faildes/merge-models`

- Make `models` file and `vae` file.  
`!mkdir models`  
`!mkdir vae`

- After installing models and vaes to the right directory,  
Run `merge.py`.   
`!cd /content/merge-models/`  
`!python merge.py...`

## Potential Problems & Troubleshooting

- Depending on your operating system and specific installation of python you might need to replace `python` with `py`, `python3`, `conda` or something else entirely.

## Credits

- [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for overall designing.
- [eyriewow](https://github.com/eyriewow/merge-models) for original merge-models.
- [lopho](https://github.com/lopho/stable-diffusion-prune) and [arenasys](https://github.com/arenasys/stable-diffusion-webui-model-toolkit) for Pruning system.
- [idelairre](https://github.com/idelairre/sd-merge-models) for Geometric, Sigmoid and Max Sum.
- [s1dlx](https://github.com/s1dlx/meh) for Multiply Difference and Similarity Add Difference.
- [hako-mikan](https://github.com/hako-mikan/sd-webui-supermerger) for Tensor Sum, Train Difference, Triple Sum, Sum twice, Smooth Add Difference, Finetuning, Cosine Merging and Elemental Merge.
- [bbc-mc](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui) for Block Weighted Merge.
- [martyn](https://github.com/martyn/safetensors-merge-supermario) for DARE Merge.
- Eyriewow got the merging logic in `merge.py` from [this post](https://discord.com/channels/1010980909568245801/1011008178957320282/1018117933894996038) by r_Sh4d0w, who seems to have gotten it from [mlfoundations/wise-ft](https://github.com/mlfoundations/wise-ft)
