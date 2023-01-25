import os
import argparse
import re
import shutil
import torch
import safetensors.torch
import math
import random
import collections
from collections import namedtuple
import gradio as gr
import modules.scripts as scripts
from gradio import interface as gr
from tqdm import tqdm
from modules import sd_models
from string import Template
from pathlib import Path
from modules.processing import process_images, fix_seed, Processed
from modules.shared import opts
from modules.paths import models_path
import modules.sd_models
from modules import shared, sd_vae, images, sd_models, devices, script_callbacks
from modules.ui_common import plaintext_to_html
import copy
import glob
from copy import deepcopy

base_vae = None
loaded_vae_file = None
checkpoint_info = None

checkpoints_loaded = collections.OrderedDict()

def get_base_vae(model):
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info and model:
        return base_vae
    return None


def store_base_vae(model):
    global base_vae, checkpoint_info
    if checkpoint_info != model.sd_checkpoint_info:
        assert not loaded_vae_file, "Trying to store non-base VAE!"
        base_vae = deepcopy(model.first_stage_model.state_dict())
        checkpoint_info = model.sd_checkpoint_info


def delete_base_vae():
    global base_vae, checkpoint_info
    base_vae = None
    checkpoint_info = None


def restore_base_vae(model):
    global loaded_vae_file
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info:
        print("Restoring base VAE")
        _load_vae_dict(model, base_vae)
        loaded_vae_file = None
    delete_base_vae()


def get_filename(filepath):
    return os.path.basename(filepath)


def refresh_vae_list():
    vae_dict.clear()

    paths = [
        os.path.join(sd_models.model_path, '**/*.vae.ckpt'),
        os.path.join(sd_models.model_path, '**/*.vae.pt'),
        os.path.join(sd_models.model_path, '**/*.vae.safetensors'),
        os.path.join(vae_path, '**/*.ckpt'),
        os.path.join(vae_path, '**/*.pt'),
        os.path.join(vae_path, '**/*.safetensors'),
    ]

    if shared.cmd_opts.ckpt_dir is not None and os.path.isdir(shared.cmd_opts.ckpt_dir):
        paths += [
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.ckpt'),
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.pt'),
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.safetensors'),
        ]

    if shared.cmd_opts.vae_dir is not None and os.path.isdir(shared.cmd_opts.vae_dir):
        paths += [
            os.path.join(shared.cmd_opts.vae_dir, '**/*.ckpt'),
            os.path.join(shared.cmd_opts.vae_dir, '**/*.pt'),
            os.path.join(shared.cmd_opts.vae_dir, '**/*.safetensors'),
        ]

    candidates = []
    for path in paths:
        candidates += glob.iglob(path, recursive=True)

    for filepath in candidates:
        name = get_filename(filepath)
        vae_dict[name] = filepath


def find_vae_near_checkpoint(checkpoint_file):
    checkpoint_path = os.path.splitext(checkpoint_file)[0]
    for vae_location in [checkpoint_path + ".vae.pt", checkpoint_path + ".vae.ckpt", checkpoint_path + ".vae.safetensors"]:
        if os.path.isfile(vae_location):
            return vae_location

    return None


def resolve_vae(checkpoint_file):
    if shared.cmd_opts.vae_path is not None:
        return shared.cmd_opts.vae_path, 'from commandline argument'

    is_automatic = shared.opts.sd_vae in {"Automatic", "auto"}  # "auto" for people with old config

    vae_near_checkpoint = find_vae_near_checkpoint(checkpoint_file)
    if vae_near_checkpoint is not None and (shared.opts.sd_vae_as_default or is_automatic):
        return vae_near_checkpoint, 'found near the checkpoint'

    if shared.opts.sd_vae == "None":
        return None, None

    vae_from_options = vae_dict.get(shared.opts.sd_vae, None)
    if vae_from_options is not None:
        return vae_from_options, 'specified in settings'

    if not is_automatic:
        print(f"Couldn't find VAE named {shared.opts.sd_vae}; using None instead")

    return None, None


def load_vae_dict(filename, map_location):
    vae_ckpt = sd_models.read_state_dict(filename, map_location=map_location)
    vae_dict_1 = {k: v for k, v in vae_ckpt.items() if k[0:4] != "loss" and k not in vae_ignore_keys}
    return vae_dict_1


def load_vae(model, vae_file=None, vae_source="from unknown source"):
    global vae_dict, loaded_vae_file
    # save_settings = False

    cache_enabled = shared.opts.sd_vae_checkpoint_cache > 0

    if vae_file:
        if cache_enabled and vae_file in checkpoints_loaded:
            # use vae checkpoint cache
            print(f"Loading VAE weights {vae_source}: cached {get_filename(vae_file)}")
            store_base_vae(model)
            _load_vae_dict(model, checkpoints_loaded[vae_file])
        else:
            assert os.path.isfile(vae_file), f"VAE {vae_source} doesn't exist: {vae_file}"
            print(f"Loading VAE weights {vae_source}: {vae_file}")
            store_base_vae(model)

            vae_dict_1 = load_vae_dict(vae_file, map_location=shared.weight_load_location)
            _load_vae_dict(model, vae_dict_1)

            if cache_enabled:
                # cache newly loaded vae
                checkpoints_loaded[vae_file] = vae_dict_1.copy()

        # clean up cache if limit is reached
        if cache_enabled:
            while len(checkpoints_loaded) > shared.opts.sd_vae_checkpoint_cache + 1: # we need to count the current model
                checkpoints_loaded.popitem(last=False)  # LRU

        # If vae used is not in dict, update it
        # It will be removed on refresh though
        vae_opt = get_filename(vae_file)
        if vae_opt not in vae_dict:
            vae_dict[vae_opt] = vae_file

    elif loaded_vae_file:
        restore_base_vae(model)

    loaded_vae_file = vae_file


# don't call this from outside
def _load_vae_dict(model, vae_dict_1):
    model.first_stage_model.load_state_dict(vae_dict_1)
    model.first_stage_model.to(devices.dtype_vae)


def clear_loaded_vae():
    global loaded_vae_file
    loaded_vae_file = None


unspecified = object()


def reload_vae_weights(sd_model=None, vae_file=unspecified):
    from modules import lowvram, devices, sd_hijack

    if not sd_model:
        sd_model = shared.sd_model

    checkpoint_info = sd_model.sd_checkpoint_info
    checkpoint_file = checkpoint_info.filename

    if vae_file == unspecified:
        vae_file, vae_source = resolve_vae(checkpoint_file)
    else:
        vae_source = "from function argument"

    if loaded_vae_file == vae_file:
        return

    if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
        lowvram.send_everything_to_cpu()
    else:
        sd_model.to(devices.cpu)

    sd_hijack.model_hijack.undo_hijack(sd_model)

    load_vae(sd_model, vae_file, vae_source)

    sd_hijack.model_hijack.hijack(sd_model)
    script_callbacks.model_loaded_callback(sd_model)

    if not shared.cmd_opts.lowvram and not shared.cmd_opts.medvram:
        sd_model.to(devices.device)

    print("VAE weights loaded.")
    return sd_model

base_dir = Path(scripts.basedir())

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

    vae_file = sd_vae.vae_dict.get(vae_file, None)
    sd_vae.load_vae(shared.sd_model, vae_file)

    print("Merge complete")
    return True


class Script(scripts.Script):
    
    def title(self):
        return "Block Model Merge Custom"

    def show(self, is_img2img):
        return True

    def get_vae_options():
        vae_options = list(sd_vae.vae_dict.keys())
        return vae_options
    
    def ui(self, is_img2img):
        
#        options = list(sd_vae.vae_dict.keys())
#        vae_file = gr.inputs.Select(options, label='VAE File', default=options[0])
        vae_path = os.path.abspath(os.path.join(models_path, "VAE"))
        vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}
        vae_dict = {}
        gpu_merge = gr.Checkbox(label="Merge using GPU", value=True, elem_id="gpu-merge")
        verbose = gr.Checkbox(label="Verbose", value=False, elem_id="verbose-merge")
        finishreload = gr.Checkbox(label="Reload checkpoint when finished", value=False, elem_id="reload-merge")
        weights = gr.Textbox(label="Weights", lines=5, max_lines=2000, elem_id="merge-weights")

        return [vae_file, gpu_merge, verbose, finishreload, weights]

    def run(self, p, vae_file, gpu_merge, verbose, finishreload, weights):
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
