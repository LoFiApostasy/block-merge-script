
import sys
sys.path.append('\\modules')
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
        dropdown = gr.Dropdown(label="Select VAE Checkpoint", choices=sd_vae.vae_dict.get(bake_in_vae, None))
        #dropdown = gr.Dropdown(label="Select VAE Checkpoint", choices=sd_vae.vae_dict())
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
        bake_in_vae_filename = sd_vae.vae_dict.get(bake_in_vae, None)
        if bake_in_vae_filename is not None:
            print(f"Baking in VAE from {bake_in_vae_filename}")
            shared.state.textinfo = 'Baking in VAE'
            vae_dict = sd_vae.load_vae_dict(bake_in_vae_filename, map_location='cpu')

            for key in vae_dict.keys():
                theta_0_key = 'first_stage_model.' + key
                if theta_0_key in theta_0:
                    theta_0[theta_0_key] = to_half(vae_dict[key], save_as_half)

            del vae_dict

    #    if save_as_half and not theta_func2:
    #        for key in theta_0.keys():
    #            theta_0[key] = to_half(theta_0[key], save_as_half)

        if discard_weights:
            regex = re.compile(discard_weights)
            for key in list(theta_0):
                if re.search(regex, key):
                    theta_0.pop(key, None)

        ckpt_dir = shared.cmd_opts.ckpt_dir or sd_models.model_path

        filename = filename_generator() if custom_name == '' else custom_name
        filename += ".inpainting" if result_is_inpainting_model else ""
        filename += "." + checkpoint_format

        output_modelname = os.path.join(ckpt_dir, filename)

        shared.state.nextjob()
        shared.state.textinfo = "Saving"
        print(f"Saving to {output_modelname}...")

        _, extension = os.path.splitext(output_modelname)
        if extension.lower() == ".safetensors":
            safetensors.torch.save_file(theta_0, output_modelname, metadata={"format": "pt"})
        else:
            torch.save(theta_0, output_modelname)

        sd_models.list_models()

        create_config(output_modelname, config_source, primary_model_info, secondary_model_info, tertiary_model_info)

        print(f"Checkpoint saved to {output_modelname}.")
        shared.state.textinfo = "Checkpoint saved"
        shared.state.end()

        return [*[gr.Dropdown.update(choices=sd_models.checkpoint_tiles()) for _ in range(4)], "Checkpoint saved to " + output_modelname]
    
#######
    def run_modelmerger(id_task, primary_model_name, secondary_model_name, tertiary_model_name, interp_method, multiplier, save_as_half, custom_name, checkpoint_format, config_source, bake_in_vae, discard_weights):
    shared.state.begin()
    shared.state.job = 'model-merge'

    def fail(message):
        shared.state.textinfo = message
        shared.state.end()
        return [*[gr.update() for _ in range(4)], message]

    def weighted_sum(theta0, theta1, alpha):
        return ((1 - alpha) * theta0) + (alpha * theta1)

    def get_difference(theta1, theta2):
        return theta1 - theta2

    def add_difference(theta0, theta1_2_diff, alpha):
        return theta0 + (alpha * theta1_2_diff)

    def filename_weighted_sum():
        a = primary_model_info.model_name
        b = secondary_model_info.model_name
        Ma = round(1 - multiplier, 2)
        Mb = round(multiplier, 2)

        return f"{Ma}({a}) + {Mb}({b})"

    def filename_add_difference():
        a = primary_model_info.model_name
        b = secondary_model_info.model_name
        c = tertiary_model_info.model_name
        M = round(multiplier, 2)

        return f"{a} + {M}({b} - {c})"

    def filename_nothing():
        return primary_model_info.model_name

    theta_funcs = {
        "Weighted sum": (filename_weighted_sum, None, weighted_sum),
        "Add difference": (filename_add_difference, get_difference, add_difference),
        "No interpolation": (filename_nothing, None, None),
    }
    filename_generator, theta_func1, theta_func2 = theta_funcs[interp_method]
    shared.state.job_count = (1 if theta_func1 else 0) + (1 if theta_func2 else 0)

    if not primary_model_name:
        return fail("Failed: Merging requires a primary model.")

    primary_model_info = sd_models.checkpoints_list[primary_model_name]

    if theta_func2 and not secondary_model_name:
        return fail("Failed: Merging requires a secondary model.")

    secondary_model_info = sd_models.checkpoints_list[secondary_model_name] if theta_func2 else None

    if theta_func1 and not tertiary_model_name:
        return fail(f"Failed: Interpolation method ({interp_method}) requires a tertiary model.")

    tertiary_model_info = sd_models.checkpoints_list[tertiary_model_name] if theta_func1 else None

    result_is_inpainting_model = False

    if theta_func2:
        shared.state.textinfo = f"Loading B"
        print(f"Loading {secondary_model_info.filename}...")
        theta_1 = sd_models.read_state_dict(secondary_model_info.filename, map_location='cpu')
    else:
        theta_1 = None

    if theta_func1:
        shared.state.textinfo = f"Loading C"
        print(f"Loading {tertiary_model_info.filename}...")
        theta_2 = sd_models.read_state_dict(tertiary_model_info.filename, map_location='cpu')

        shared.state.textinfo = 'Merging B and C'
        shared.state.sampling_steps = len(theta_1.keys())
        for key in tqdm.tqdm(theta_1.keys()):
            if key in checkpoint_dict_skip_on_merge:
                continue

            if 'model' in key:
                if key in theta_2:
                    t2 = theta_2.get(key, torch.zeros_like(theta_1[key]))
                    theta_1[key] = theta_func1(theta_1[key], t2)
                else:
                    theta_1[key] = torch.zeros_like(theta_1[key])

            shared.state.sampling_step += 1
        del theta_2

        shared.state.nextjob()

    shared.state.textinfo = f"Loading {primary_model_info.filename}..."
    print(f"Loading {primary_model_info.filename}...")
    theta_0 = sd_models.read_state_dict(primary_model_info.filename, map_location='cpu')

    print("Merging...")
    shared.state.textinfo = 'Merging A and B'
    shared.state.sampling_steps = len(theta_0.keys())
    for key in tqdm.tqdm(theta_0.keys()):
        if theta_1 and 'model' in key and key in theta_1:

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

                assert a.shape[1] == 9 and b.shape[1] == 4, f"Bad dimensions for merged layer {key}: A={a.shape}, B={b.shape}"

                theta_0[key][:, 0:4, :, :] = theta_func2(a[:, 0:4, :, :], b, multiplier)
                result_is_inpainting_model = True
            else:
                theta_0[key] = theta_func2(a, b, multiplier)

            theta_0[key] = to_half(theta_0[key], save_as_half)

        shared.state.sampling_step += 1

    del theta_1
#end new code
