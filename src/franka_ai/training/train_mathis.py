from pathlib import Path
import numpy as np
import time
import tyro # Import tyro
from dataclasses import dataclass, field # Nécessaire pour tyro
import sys
import os
import json
import warnings
from typing import List, Optional
import wandb
import torch
from torch.utils.data import DataLoader

from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType
# from deepil.training.utils import LeRobotDatasetMetadata, MultiLeRobotDatasetPatch
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from franka_ai.dataset.utils import LeRobotDatasetPatch, MultiLeRobotDatasetPatch
from franka_ai.models.actPatch.modeling_act_mathis import ACTPolicyPatch
from franka_ai.models.actPatch.configuration_act_mathis import ACTConfigPatch



"""
Run the code: 

python src/franka_ai/training/train_mathis.py --algorithm act



"""



# -------------------------------------------------------------
# Ajout explicite du répertoire parent du script (deepil) au sys.path
# -------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)
# -------------------------------------------------------------

# --- Tyro Configuration ---
@dataclass
class Args:
    """Configuration pour l'entraînement d'une politique off-policy."""
    names_dataset: List[str] = field(default_factory=lambda: ["single_outliers"], metadata={"help": "Nom(s) du (des) dataset(s) à utiliser."})
    num_training_demos: List[Optional[int]] = field(default_factory=lambda: [1], metadata={"help": "Nombre de démos à utiliser pour l'entraînement par dataset. 'None' pour toutes."})
    device: str = field(default="cuda", metadata={"help": "Dispositif d'entraînement (e.g., 'cuda', 'cpu')."})
    algorithm: str = field(default="dp", metadata={"help": "Algorithme à utiliser : 'dp' (Diffusion Policy), 'flow' (Flow Policy), 'act' (ACT Policy)."})
    percentage_val: float = field(default=10.0, metadata={"help": "Pourcentage du dataset à utiliser pour la validation (e.g., 10.0 pour 10%)."})
    horizon: int = field(default=50, metadata={"help": "Taille du chunk d'actions."})
    
    # Hyperparamètres d'entraînement
    batch_size: int = field(default=16, metadata={"help": "Taille de lot pour l'entraînement."})
    val_batch_size: int = field(default=64, metadata={"help": "Taille de lot pour la validation."})
    training_steps: int = field(default=4000, metadata={"help": "Nombre total d'étapes d'entraînement (steps)."})
    learning_rate: float = field(default=1e-5, metadata={"help": "Taux d'apprentissage de l'optimiseur. Sera écrasé par cfg.optimizer_lr si défini."})
    patience: int = field(default=1000000, metadata={"help": "Patience en époques (ou steps si une epoch est rapide) avant d'arrêter si le val_loss n'améliore pas."})
    step_log_freq: int = field(default=200, metadata={"help": "Fréquence de logging et d'évaluation (en steps)."})
    step_chkpt: int = field(default=1000, metadata={"help": "Fréquence de checkpoint (en steps)."})
    epoch_log_freq: int = field(default=100000, metadata={"help": "Fréquence de logging/sauvegarde supplémentaire à la fin de l'époque (en étapes), si non couvert par step_log_freq."})
    num_workers: int = field(default=4, metadata={"help": "Nombre de workers pour les DataLoaders."})


# --- Fonctions Utilitaires ---

def split_train_val(arr: np.ndarray, val_percentage: float) -> tuple[List[int], List[int]]:
    """Divise les indices d'épisodes en ensembles d'entraînement et de validation."""
    n = arr.shape[0]
    p = int(val_percentage * n / 100.0) # Ajuster la formule pour le pourcentage
    
    # S'assurer d'avoir au moins 1 épisode de validation si possible
    if p == 0 and n > 0 and val_percentage > 0:
         p = 1
    
    if p >= n:
        warnings.warn(f"Le pourcentage de validation est trop élevé. {p} >= {n}. La validation sera vide.")
        return arr.tolist(), []


    choice = np.random.choice(range(n), size=(p,), replace=False)
    ind = np.zeros(n, dtype=bool)
    ind[choice] = True
    rest = ~ind
    return arr[rest].tolist(), arr[ind].tolist()

def prepare_batch(batch: dict, device: torch.device) -> dict:
    """Déplace les tenseurs du lot vers le device et applique des traitements spécifiques."""
    new_batch = {}
    for k, v in batch.items():
        # Ignorer 'task' qui n'est pas toujours un tenseur
        if k != "task":
            new_batch[k] = v.to(device, non_blocking=True)
        
        # Traitement spécifique pour 'depth' si présent
        if "depth" in k:
            # Assumer que la dimension de canal est la 4ème (après batch, temps, hauteur, largeur)
            # et la réduire à [..., 1]
            new_batch[k] = new_batch[k][..., 0, None]


        # temp test
        if k == "observation.images.front_cam1":
            new_batch[k] = torch.zeros_like(new_batch[k])
            
    return new_batch

def evaluate(policy, valloader: DataLoader, device: torch.device) -> float:
    """Évalue la politique sur l'ensemble de validation et retourne la perte moyenne."""
    policy.eval()
    val_loss, val_steps = 0.0, 0
    with torch.no_grad():
        for valbatch in valloader:
            valbatch = prepare_batch(valbatch, device)
            loss, _ = policy.forward(valbatch)
            val_loss += loss.item()
            val_steps += 1
    
    if val_steps == 0:
        return np.inf # Éviter la division par zéro si le valloader est vide
    
    return val_loss / val_steps


def main():


    dataset_path_root = f"/mnt/Data/datasets/lerobot"



    # --- 1. Charger la Configuration avec tyro ---
    args = tyro.cli(Args)

    print("Configuration d'entraînement:")
    for key, value in args.__dict__.items():
        print(f"  {key}: {value}")

    total_name = "_".join(args.names_dataset)+"_".join([str(num) for num in args.num_training_demos]) + "_"
    
    output_directory = Path(f"./outputs/train/{total_name}_{args.algorithm}_{int(args.percentage_val)}_{args.horizon}")
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # --- 2. Préparation des Datasets ---
    num_episodes, train_episodes, val_episodes = {}, {}, {}
    for i, ndataset in enumerate(args.names_dataset):
        dataset_path = Path(dataset_path_root) / ndataset
        info_path = dataset_path / "meta" / "info.json"
        
        with open(info_path, "r") as reader:
            num_episodes[ndataset] = json.load(reader)["total_episodes"]

        episodes_non_avoided = list(range(num_episodes[ndataset]))
        split_path = dataset_path / "meta" / "split.json"
        if split_path.exists():
            with open(split_path, "r") as reader:
                dico = json.load(reader)
                episodes_non_avoided = [ep for ep in episodes_non_avoided if ep not in dico.get("avoid", [])]
        
        train_episodes[ndataset], val_episodes[ndataset] = split_train_val(np.array(episodes_non_avoided), args.percentage_val)

        if i < len(args.num_training_demos) and args.num_training_demos[i] is not None:
            num_demos_requested = args.num_training_demos[i]
            if num_demos_requested > len(train_episodes[ndataset]):
                print(f"Warning : le nombre de démos d'entraînement est trop élevé pour {ndataset}. Utilisation du max: {len(train_episodes[ndataset])}.")
            else:
                train_episodes[ndataset] = train_episodes[ndataset][:num_demos_requested]

    # Définition du device
    device = torch.device(args.device)

    # --- 3. Configuration de la Politique ---
    # Nous assumons qu'il n'y a qu'un seul dataset pour les métadonnées
    # dataset_metadata = LeRobotDatasetMetadata("", root=f"./datasets/{args.names_dataset[0]}")
    dataset_metadata = LeRobotDatasetMetadata("", root=f"{dataset_path_root}/{args.names_dataset[0]}")
    print(f"{dataset_path_root}/{args.names_dataset[0]}")
    
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key in ["observation.state", "observation.images.front_cam1"]} #if ((key not in output_features) and ("gripper_cam" not in key))}

    print(input_features)
    
    cfg = None
    horizon = 0
    if args.algorithm == "flow":
        cfg = ACTConfigPatch(input_features=input_features, output_features=output_features, resize_shape=(192, 144))
        #cfg = FlowConfig(input_features=input_features, output_features=output_features, resize_shape=(192, 144), vision_backbone="vit_b_16", pretrained_backbone_weights="ViT_B_16_Weights.IMAGENET1K_V1")
        # Suppression des crops aléatoires pour le Flow Policy si ce n'est pas nécessaire
        cfg.crop_pos = None
        cfg.random_crop_shape = None 
        horizon = cfg.horizon
    elif args.algorithm == "dp":
        cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
        horizon = cfg.horizon
    elif args.algorithm == "act":
        cfg = ACTConfigPatch(input_features=input_features, output_features=output_features, resize_shape=(192, 144), n_action_steps=args.horizon, chunk_size=args.horizon)
        # Suppression des crops aléatoires pour l'ACT
        cfg.crop_pos = None
        cfg.random_crop_shape = None
        horizon = cfg.chunk_size
    else:
        raise ValueError(f"Algorithme inconnu: {args.algorithm}")

    # Le taux d'apprentissage de tyro prend le pas sur la valeur par défaut de la config ACT/Diffusion/Flow
    # si la config par défaut est utilisée. 
    # Sinon, vous pouvez décommenter la ligne suivante si vous voulez forcer le LR de tyro.
    # cfg.optimizer_lr = args.learning_rate 

    delta_timestamps = {k:[-(cfg.n_obs_steps - 1 - i) / dataset_metadata.fps for i in range(cfg.n_obs_steps)] for k in input_features.keys()}
    for output_key in output_features.keys():
        delta_timestamps[output_key] = [i / dataset_metadata.fps for i in range(horizon)]

    # # --- 4. Chargement des Données et DataLoaders ---
    # # Gérer la logique MultiLeRobotDataset si plusieurs datasets, mais pour l'instant, on se concentre sur un seul dataset.
    # train_episodes_list = train_episodes[args.names_dataset[0]]
    # val_episodes_list = val_episodes[args.names_dataset[0]]

    #dataset = LeRobotDatasetPatch("", root=f"./datasets/{args.names_dataset[0]}", delta_timestamps=delta_timestamps, episodes=train_episodes_list, tolerance_s=1000)
    #valset = LeRobotDatasetPatch("", root=f"./datasets/{args.names_dataset[0]}", delta_timestamps=delta_timestamps, episodes=val_episodes_list, tolerance_s=1000)

    dataset = MultiLeRobotDatasetPatch(args.names_dataset, root=f"{dataset_path_root}", delta_timestamps=delta_timestamps, episodes=train_episodes, tolerances_s={name:1000 for name in train_episodes.keys()})

    empty_val = False
    for key in val_episodes.keys():
        if len(val_episodes[key]) == 0:
            empty_val = True
            break

    if empty_val == False:
        valset = MultiLeRobotDatasetPatch(args.names_dataset, root=f"{dataset_path_root}", delta_timestamps=delta_timestamps, episodes=val_episodes, tolerances_s={name:1000 for name in train_episodes.keys()})

    dataloader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        prefetch_factor=2,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    if empty_val == False:
        valloader = DataLoader(
            valset,
            num_workers=args.num_workers,
            batch_size=args.val_batch_size,
            shuffle=False, # Pas besoin de shuffle pour la validation
            pin_memory=device.type != "cpu",
            drop_last=False, # Garder tous les exemples de validation
        )

    # --- 5. Initialisation de la Politique et de l'Optimiseur ---
    if args.algorithm == "flow":
        policy = ACTPolicyPatch(cfg, dataset_stats=dataset.stats)
        #policy = FlowPolicy(cfg, dataset_stats=stats)
    elif args.algorithm == "dp":
        policy = DiffusionPolicy(cfg, dataset_stats=dataset.stats)
    elif args.algorithm == "act":
        policy = ACTPolicyPatch(cfg, dataset_stats=dataset.stats)

    policy.train()
    policy.to(device=device)
    
    # Utiliser le LR de la config, qui est maintenant args.learning_rate si on a décommenté
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.optimizer_lr, weight_decay=cfg.optimizer_weight_decay)
    # optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer_lr, betas=(0.9, 0.999), weight_decay=cfg.optimizer_weight_decay)


    # --- 6. Boucle d'Entraînement ---
    train_stats = [["epoch"], ["step"], ["train_loss"], ["val_loss"]]
    
    def append_stats(stats, epoch, step, train_loss, val_loss):
        stats[0].append(epoch)
        stats[1].append(step)
        stats[2].append(train_loss)
        stats[3].append(val_loss)
        np.savetxt(output_directory / 'training_log.csv', [p for p in zip(*stats)], delimiter=',', fmt='%s')

    t_start = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        step = 0
        best_val_loss = np.inf
        epoch, epoch_since_best = 0, 0
        done = False
        
        while not done:
            total_loss_train_period = 0.0
            
            for batch in dataloader:
                
                # --- Étape d'entraînement ---
                batch = prepare_batch(batch, device)

                #print({k:v.shape for k,v in batch.items() if k != "task"})

                loss, _ = policy.forward(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss_train_period += loss.item()
                
                # --- Logging et Évaluation Périodique (en steps) ---
                if (step + 1) % args.step_log_freq == 0:
                    
                    # 1. Calcul des pertes
                    avg_train_loss = total_loss_train_period / args.step_log_freq
                    val_loss = evaluate(policy, valloader, device)
                    
                    print(f"epoch: {epoch} step: {step+1} train loss: {avg_train_loss:.6f} val loss: {val_loss:.6f}")
                    append_stats(train_stats, epoch, step+1, avg_train_loss, val_loss)
                    
                    # 2. Sauvegarde du meilleur modèle
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        policy.save_pretrained(Path(str(output_directory) + "_best"))
                        epoch_since_best = 0
                    
                    # 3. Préparation pour la prochaine période
                    policy.train()
                    total_loss_train_period = 0.0

                if (step + 1) % args.step_chkpt == 0:
                    policy.save_pretrained(Path(str(output_directory) + "_" + str(step)))

                step += 1
                if step >= args.training_steps:
                    done = True
                    break

            # --- Logging et Évaluation de Fin d'Époque (si non couverte par le step_log_freq) ---
            if not done and (epoch + 1) % (args.epoch_log_freq / len(dataloader)) == 0 and step % args.step_log_freq != 0 and (empty_val == False):
                val_loss = evaluate(policy, valloader, device)
                
                # Le total_loss_train_period peut contenir des pertes partielles pour une période
                # incomplète, nous utilisons la perte de validation pour la comparaison
                print(f"Fin d'epoch {epoch}: step: {step} val loss: {val_loss:.6f}")
                
                # Enregistrer l'évaluation de l'époque
                append_stats(train_stats, epoch, step, np.nan, val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    policy.save_pretrained(Path(str(output_directory) + "_best"))
                    epoch_since_best = 0
                    
                policy.train()
            
            epoch += 1
            epoch_since_best += 1
            
            # --- Condition d'arrêt ---
            if epoch_since_best * len(dataloader) >= args.patience:
                print(f"Arrêt précoce : aucune amélioration depuis {args.patience} steps.")
                done = True

    print(f"Temps total d'entraînement: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()