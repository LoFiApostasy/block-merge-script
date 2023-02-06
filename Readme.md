Small update to fix vae selection breaking the script.

# Merge Block Weighted - Script

- This is an extension for [AUTOMATIC1111's Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- Based on the following prior work:
    - Merge Block Weighted https://note.com/kohya_ss/n/n9a485a066d5b idea by kohya_ss
    - Merge Block Weighted - GUI https://github.com/bbc-mc/sdweb-merge-block-weighted-gui

- This script does the same thing as sdweb-merge-block-weighted-gui. But the resulting merge is only used to generate the current prompt.
- The weights field has multiple lines. A set of weights can be specified on each line and a merge will be performed and the prompt run on the resulting merge. The merged model is discarded once it's done generating the prompt (I suggest also getting sdweb-merge-block-weighted-gui to save merges as checkpoints)
- The weights used are saved into the metadata of generated images
- Each line in the weights field must follow the format "model1,model2,base_alpha,in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11,mid,out11,out10,out9,out8,out7,out6,out5,out4,out3,out2,out1,out0"
    - e.g. "dogModel,artModel,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4"
        - would make a merge that's 40% dogModel and 60% artModel
    - e.g. "dogModel,artModel,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0"
        -   would make a merge that's all dog model except output layers 5, 4, and 3 which are artModel
- These values can be copied into the sdweb-merge-block-weighted-gui extension to create a checkpoint of the merge

## How to Install

- Go to `Extensions` tab on your web UI
- `Install from URL` with this repo URL
- Install

####################################################################################

A wonderful writeup on Blockmerge exists here: https://rentry.org/BlockMergeExplained

########### Examples ###########

Here is a quick script input to get an idea of whats going on in Model2

Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0
Model1.ckpt,Model2.safetensors,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1



Here is another script 2 batch process.
Model1 and Model2 mixed at 50%. 
First batch is to strengthen the concept your pulling from model2, note that I added a divider run with .9,.2 so its easy to find the last picture of the batch.
After several batches I will open my block merge with all sliders set to 50% and incriment the sliders by 1 that represent a positive improvement, and 2 forna huge improvement. After several batche runs it becomes clear which ones are the sliders that contain the concepts.
keep in mind when doing this that the picture order is:
(no change),IN00 - IN11, MID00, OUT11 - OUT00

Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,1,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,1,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,1,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,1,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,1,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,1,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,1,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,1,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,1,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,1
Model1.ckpt,Model2.safetensors,.9,.2,.9,.2,.9,.2,.9,.2,.9,.2,.9,.2,.9,.2,.9,.2,.9,.2,.9,.2,.9,.2,.9,.2,.9,.2


After setting the sliders to the new stronger concept, then run the reverse, subtracting model2 and noting reduced deformities.
### Note, this part is an example because you would have set your sliders to your improved concept.  ###
### This is here to demonstrate the pattern. ###

Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,0,.5,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,0,.5,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,0,.5,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,0,.5,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,0,.5,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,0,.5,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,0,.5,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,0,.5,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,0,.5
Model1.ckpt,Model2.safetensors,1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,0
Model1.ckpt,Model2.safetensors,.9,.2,.9,.2,.9,.2,.9,.2,.9,.2,.9,.2,.9,.2,.9,.2,.9,.2,.9,.2,.9,.2,.9,.2,.9,.2
