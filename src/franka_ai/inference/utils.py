from datetime import datetime
import os
import yaml


# TODO:
# 1) magari in input setup_folders accetta policy name o percorso di caricamento pesi dall utente?


def get_configs_inference(config_rel_path):

    # set path
    path = os.path.join(os.getcwd(), config_rel_path)

    # open config
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    inference_cfg = cfg["inference"]

    return inference_cfg


# def setup_folder():

#     """Create output folder structure for evaluation runs."""
    
#     # Base folder for all experiments
#     base_output_dir = os.path.join(os.getcwd(), "outputs")

#     # Unique run name
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     # run_name = f"{timestamp}_{cfg.__class__.__name__.lower()}_{dataset_id.split('/')[-1]}"
#     # output_dir = os.path.join(base_output_dir, run_name)

#     # Subfolders
#     eval_dir = os.path.join(base_output_dir, "evals", timestamp)

#     os.makedirs(eval_dir, exist_ok=True)

#     return eval_dir
    