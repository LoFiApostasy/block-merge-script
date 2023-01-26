

import os
import argparse
import re
import torch
import math
import random
import gradio as gr
import modules.scripts as scripts
from tqdm import tqdm
from modules import sd_models
from string import Template
from pathlib import Path
from modules.processing import process_images, fix_seed, Processed
from modules.shared import opts
import modules.sd_models
from modules import shared, sd_vae, images, sd_models #new code, last 2
import copy

#new code
import shutil
from modules.ui_common import plaintext_to_html
import safetensors.torch
#end new code

base_dir = Path(scripts.basedir())

#new code
def create_config(ckpt_result, config_source, a, b, c):
    def config(x):
        res = sd_models.find_checkpoint_config(x) if x else None
        return res if res != shared.sd_default_config else None

    if config_source == 0:
        cfg = config(a) or config(b) or config(c)
    elif config_source == 1:
        cfg = config(b)
    elif config_source == 2:
        cfg = config(c)
    else:
        cfg = None

    if cfg is None:
        return

    filename, _ = os.path.splitext(ckpt_result)
    checkpoint_filename = filename + ".yaml"

    print("Copying config:")
    print("   from:", cfg)
    print("     to:", checkpoint_filename)
    shutil.copyfile(cfg, checkpoint_filename)


checkpoint_dict_skip_on_merge = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]
#end new code

def apply_checkpoint(x):
    info = modules.sd_models.get_closet_checkpoint_match(x)
    if info is None:
        raise RuntimeError(f"Unknown checkpoint: {x}")
    modules.sd_models.reload_model_weights(shared.sd_model, info)

NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

def dprint(str, flg):
    if flg:
        print(str)

def merge(weights:list, model_0, model_1, device="cpu", base_alpha=0.5, verbose=False):
    if weights is None:
        weights = None
    else:
        weights = [float(w) for w in weights.split(',')]
    if len(weights) != NUM_TOTAL_BLOCKS:
        _err_msg = f"weights value must be {NUM_TOTAL_BLOCKS}."
        print(_err_msg)
        print(weights)
        return False, _err_msg

    sd_vae.delete_base_vae()
    sd_vae.clear_loaded_vae()

    device = device if device in ["cpu", "cuda"] else "cpu"

    def load_model(_model, _device):
        model_info = sd_models.get_closet_checkpoint_match(_model)
        if model_info:
            model_file = model_info.filename
        return sd_models.read_state_dict(model_file, map_location=_device)

    print("loading", model_0)
    theta_0 = load_model(model_0, device)

    print("loading", model_1)
    theta_1 = load_model(model_1, device)

    alpha = base_alpha

    re_inp = re.compile(r'\.input_blocks\.(\d+)\.')  # 12
    re_mid = re.compile(r'\.middle_block\.(\d+)\.')  # 1
    re_out = re.compile(r'\.output_blocks\.(\d+)\.') # 12

    dprint(f"-- start Stage 1/2 --", verbose)
    count_target_of_basealpha = 0
    for key in (tqdm(theta_0.keys(), desc="Stage 1/2") if not verbose else theta_0.keys()):
        if "model" in key and key in theta_1:
            dprint(f"  key : {key}", verbose)
            current_alpha = alpha

            # check weighted and U-Net or not
            if weights is not None and 'model.diffusion_model.' in key:
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
                    print(f"error. illegal block index: {key}")
                    return False, ""
                if weight_index >= 0:
                    current_alpha = weights[weight_index]
                    dprint(f"weighted '{key}': {current_alpha}", verbose)
            else:
                count_target_of_basealpha = count_target_of_basealpha + 1
                dprint(f"base_alpha applied: [{key}]", verbose)

            theta_0[key] = (1 - current_alpha) * theta_0[key] + current_alpha * theta_1[key]

        else:
            dprint(f"  key - {key}", verbose)

    dprint(f"-- start Stage 2/2 --", verbose)
    for key in tqdm(theta_1.keys(), desc="Stage 2/2"):
        if "model" in key and key not in theta_0:
            dprint(f"  key : {key}", verbose)
            theta_0.update({key:theta_1[key]})
        else:
            dprint(f"  key - {key}", verbose)

    print("Swapping sd_model...")

    shared.sd_model.load_state_dict(theta_0, strict=False)

    print("Restoring vae...")

#    vae_file = sd_vae.get_vae_from_settings()
#    sd_vae.load_vae(shared.sd_model, vae_file)
#new code
    vae_file = dropdown
    sd_vae.load_vae(shared.sd_model, vae_file)
#end new code    

    print("Merge complete")
    return True


class Script(scripts.Script):
    def title(self):
        return "Block Model Merge Development"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        gpu_merge = gr.Checkbox(label="Merge using GPU", value=True, elem_id="gpu-merge")
        verbose = gr.Checkbox(label="Verbose", value=False, elem_id="verbose-merge")
        finishreload = gr.Checkbox(label="Reload checkpoint when finished", value=False, elem_id="reload-merge")
        #new code
        dropdown = gr.Dropdown(label="Select VAE Checkpoint", choices=sd_vae.vae_dict())
        #dropdown = gr.Dropdown(label="Select VAE Checkpoint", choices=sd_models.checkpoint_tiles())
        #end new code
        weights = gr.Textbox(label="Weights", lines=5, max_lines=2000, elem_id="merge-weights")

        return [gpu_merge, verbose, finishreload, weights]#, dropdown]
    
    #new code
    bake_in_vae_filename = sd_vae.vae_dict.get(bake_in_vae, None)
    if bake_in_vae_filename is not None:
        print(f"Baking in VAE from {bake_in_vae_filename}")
        shared.state.textinfo = 'Baking in VAE'
        vae_dict = sd_vae.load_vae_dict(bake_in_vae_filename, map_location='cpu')
    #end new code
        
        for key in vae_dict.keys():
            theta_0_key = 'first_stage_model.' + key
            if theta_0_key in theta_0:
                theta_0[theta_0_key] = to_half(vae_dict[key], save_as_half)

        del vae_dict
    
    def run(self, p, gpu_merge, verbose, finishreload, weights):
        print("Running block model merge")
        output_images = []
        infotexts = []
        if p.seed == -1:
            p.seed = random.randint(0,2147483647)

        device = "cuda" if gpu_merge else "cpu"

        weights1 = [str(w) for w in weights.split('\n')]

        index = 0
        for line in weights1:
            index = index + 1
            weights2 = [str(w) for w in line.split(',')]
            if len(weights2) != 28:
                _err_msg = f"weights value on line " + str(index) + " must be \"model1, model2, base_alpha, in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, mid, out11, out10, out9, out8, out7, out6, out5, out4, out3, out2, out1, out0\"."
                print(_err_msg)
                return Processed(p, output_images, p.seed, infotexts=infotexts)

        for line in tqdm(weights1, desc="Total Progress"):
            weights2 = [str(w) for w in line.split(',')]
            model1 = weights2[0]
            model2 = weights2[1]
            base_alpha=float(weights2[2])
            unet=",".join(weights2[3:])
            p.extra_generation_params["Block Merge Weights"] = line
            if merge(unet, model1, model2, device, base_alpha, verbose):
                proc = process_images(p)
                output_images += proc.images
                infotexts += proc.infotexts

        res = Processed(p, output_images, p.seed, infotexts=infotexts)

        if finishreload:
            modules.sd_models.load_model()

        return res

#new code    

