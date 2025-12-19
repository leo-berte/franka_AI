import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

# Imports de votre projet
# from deepil.training.utils import LeRobotDatasetMetadata, MultiLeRobotDatasetPatch
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from franka_ai.dataset.utils import LeRobotDatasetPatch
from franka_ai.models.actPatch.modeling_act_mathis import ACTPolicyPatch
# from franka_ai.models.actPatch.modeling_act_original import ACTPolicy


"""
Run the code: 

python src/franka_ai/inference/evaluate_mathis.py

"""


def prepare_batch(batch: dict, device: torch.device) -> dict:
    """Déplace les tenseurs du lot vers le device et applique des traitements spécifiques."""
    new_batch = {}
    action_gt = batch["action"][:, 0, :]
    for k, v in batch.items():
        # Ignorer 'task' qui n'est pas toujours un tenseur
        if k != "task":
            new_batch[k] = v.to(device, non_blocking=True)
            #print(k, v.shape)
        
        # Traitement spécifique pour 'depth' si présent
        if "depth" in k:
            # Assumer que la dimension de canal est la 4ème (après batch, temps, hauteur, largeur)
            # et la réduire à [..., 1]
            new_batch[k] = new_batch[k][..., 0, None]

        # temp test
        if k == "observation.images.front_cam1":
            new_batch[k] = torch.zeros_like(new_batch[k])

    new_batch.pop("action")
            
    return new_batch, action_gt

def main():
    # --- 1. Configuration ---
    # Remplacez par le chemin de votre modèle sauvegardé
    # model_path = "./outputs/checkpoints/single_outliers_act_2025-12-18_16-45-49/best_model.pt"
    model_path = "./outputs/train/single_outliers1__act_10_50_best_adamw" 
    dataset_path = f"/mnt/Data/datasets/lerobot/single_outliers"

    # dataset_name = "single_outliers"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 2. Chargement de la Politique ---
    # On détecte le type dans la config ou on tente ACTPolicyPatch par défaut
    print(f"Chargement de la politique depuis {model_path}...")
    policy = ACTPolicyPatch.from_pretrained(model_path)
    policy.to(device)
    policy.eval()

    # --- 3. Chargement de l'épisode 0 du Dataset ---
    dataset_metadata = LeRobotDatasetMetadata("", root=dataset_path)
    fps = dataset_metadata.fps
    
    # On force l'utilisation de l'épisode 0 uniquement
    # episodes = {dataset_name: [0]}
    
    # On récupère les timestamps delta nécessaires pour la politique (cfg)
    cfg = policy.config
    delta_timestamps = {
        k: [-(cfg.n_obs_steps - 1 - i) / fps for i in range(cfg.n_obs_steps)] 
        for k in cfg.input_features.keys()
    }
    for output_key in cfg.output_features.keys():
        delta_timestamps[output_key] = [i / dataset_metadata.fps for i in range(cfg.chunk_size)]
    
    test_dataset = LeRobotDatasetPatch(
        repo_id=None,
        root=dataset_path,   
        delta_timestamps=delta_timestamps, 
        episodes=[0]
    )
    
    # On utilise un DataLoader sans shuffle pour garder l'ordre temporel
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_actions = []
    dataset_actions = []

    print(f"Analyse de l'épisode 0 ({len(test_dataset)} frames)...")

    # --- 4. Inférence et Collecte ---
    with torch.no_grad():
        for batch in dataloader:
            # Préparation du batch pour la politique
            # (prepare_batch déplace déjà sur le device et gère la profondeur)
            batch, action_gt = prepare_batch(batch, device)
            
            # Prédiction du modèle
            # select_action retourne souvent le chunk entier, on prend la première action [B, T, D] -> [D]

            # print(torch.linalg.norm(batch["observation.images.front_cam1"]))
            action_pred = policy.select_action(batch)

            model_actions.append(action_pred.cpu().numpy().flatten())
            dataset_actions.append(action_gt.cpu().numpy().flatten())

    model_actions = np.array(model_actions)
    dataset_actions = np.array(dataset_actions)

    # --- 5. Visualisation ---
    num_dims = model_actions.shape[1]
    fig, axes = plt.subplots(num_dims, 1, figsize=(12, 2 * num_dims), sharex=True)
    if num_dims == 1: axes = [axes]

    for i in range(num_dims):
        axes[i].plot(dataset_actions[:, i], label="Dataset (Ground Truth)", color="blue", linestyle="--", alpha=0.7)
        axes[i].plot(model_actions[:, i], label="Modèle (Prédiction)", color="red", alpha=0.8)
        axes[i].set_ylabel(f"Dim {i}")
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend()

    axes[-1].set_xlabel("Timesteps (Frames)")
    plt.suptitle(f"Comparaison Actions : Modèle vs Dataset (Épisode 0)\n{model_path}")
    plt.tight_layout()
    
    # Sauvegarde du graphique
    plot_path = model_path + "/" + "comparison_plot.png"
    plt.savefig(plot_path)
    print(f"Graphique de comparaison sauvegardé dans : {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()