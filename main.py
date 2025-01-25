import subprocess
import os


commands = [
    "clear",
    "cd priorMDM",
    "/home/ML_courses/03683533_2024/anton_kfir_daniel/conda/miniconda3/envs/priorMDM_env/bin/python -m priorMDM.sample.finetuned_motion_control --model_path save/root_horizontal_finetuned/model000280000.pt --text_condition 'a person is raising hands'"
    ]
    

import subprocess
import os

commands = [
    "clear",
    "cd priorMDM",
    "/home/ML_courses/03683533_2024/anton_kfir_daniel/conda/miniconda3/envs/priorMDM_env/bin/python -m sample.finetuned_motion_control --model_path save/root_horizontal_finetuned/model000280000.pt --text_condition 'a person is raising hands'"
]

for cmd in commands:
    try:
        if cmd == "clear":
            os.system(cmd)

        elif cmd.startswith("cd "):
            # Extract the directory to change into
            directory = cmd.split(" ", 1)[1]
            os.chdir(directory)
            print(f"Changed working directory to: {os.getcwd()}")

        else:
            print(f"Executing: {cmd}")
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"Output:\n{result.stdout}")

    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Error output:\n{e.stderr}")
        break
    except FileNotFoundError as e:
        print(f"Directory not found: {e}")
        break


 