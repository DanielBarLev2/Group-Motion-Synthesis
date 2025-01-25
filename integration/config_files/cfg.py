import os

CWD = "/home/ML_courses/03683533_2024/anton_kfir_daniel/priorMDM-Trace"

SAVE_DIR = os.path.join(CWD, "integration/save")
SCENE_KEY = "scene_000378_orca_maps_31"
DATA_PATH = os.path.join(CWD,"trace/out/orca_mixed_out/orca_map_open_loop_target_pos/data.hdf5")
MAX_FRAMES = 196
OUTPUTS_DIR = os.path.join(CWD, "integration/outputs/plots")