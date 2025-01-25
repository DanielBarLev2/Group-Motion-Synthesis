#!/bin/bash

# Clear the screen
clear

/home/ML_courses/03683533_2024/anton_kfir_daniel/conda/miniconda3/envs/trace_2/bin/python -m integration.src.generate_traces


# Navigate to the priorMDM directory
cd priorMDM || exit

# Run the Python command
/home/ML_courses/03683533_2024/anton_kfir_daniel/conda/miniconda3/envs/priorMDM_env/bin/python -m sample.finetuned_motion_control \
    --model_path save/root_horizontal_finetuned/model000280000.pt \
    --text_condition "a person is raising hands"