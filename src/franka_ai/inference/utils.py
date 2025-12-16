from datetime import datetime
import os
import yaml


# ---------
# FUNCTIONS
# ---------

def get_configs_inference(config_rel_path):

    # set path
    path = os.path.join(os.getcwd(), config_rel_path)

    # open config
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    inference_cfg = cfg["inference"]

    return inference_cfg
    