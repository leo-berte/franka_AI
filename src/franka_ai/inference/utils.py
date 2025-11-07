from datetime import datetime
import os


# TODO:
# 1) magari in input setup_folders accetta policy name o percorso di caricamento pesi dall utente?


def setup_folder():

    """Create output folder structure for evaluation runs."""
    
    # Base folder for all experiments
    base_output_dir = os.path.join(os.getcwd(), "outputs")

    # Unique run name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # run_name = f"{timestamp}_{cfg.__class__.__name__.lower()}_{dataset_id.split('/')[-1]}"
    # output_dir = os.path.join(base_output_dir, run_name)

    # Subfolders
    eval_dir = os.path.join(base_output_dir, "evals", timestamp)

    os.makedirs(eval_dir, exist_ok=True)

    return eval_dir
    