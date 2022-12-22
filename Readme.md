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