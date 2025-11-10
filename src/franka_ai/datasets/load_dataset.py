from torch.utils.data import Dataset, DataLoader, random_split
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from pprint import pprint
import torch

from franka_ai.datasets.transforms import CustomTransforms
from franka_ai.datasets.utils import build_delta_timestamps
from franka_ai.utils.seed_everything import make_worker_init_fn



# TODO:  
# 1) capire dove mettere transf per abs/rel poses + orientations
# 2) nel training vengono normalizzate features, dove/chi calcola mean/std per ogni feature?
# 3) come rimuovere depth camera dal batch, e da policy? per ora carica tutto
# 3) e se voglio aggiungere/modificare new features (es usare axis angles isntead of quaternions?)


class TransformedDataset(Dataset):
    
    """
    Wrap a base dataset and apply custom transformations to each sample.

    Args:
        base_dataset: original dataset
        custom_transforms: transformation pipeline applied to each sample
    """

    def __init__(self, base_dataset, custom_transforms):
        self.base = base_dataset
        self.custom_transforms = custom_transforms

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        return self.custom_transforms.transform(sample)
    

def make_dataloader(
    repo_id=None, 
    local_root=None, 
    visual_obs_names=None,
    device=torch.device("cpu"),
    batch_size=32,
    shuffle=True,
    train_split=0.8, 
    num_workers=4, 
    seed_val=None, 
    N_history=16,  
    N_chunk=8, 
    fps_sampling=None, 
    print_ds_info=False 
):
    
    """
    Build PyTorch DataLoaders for training and validation from a LeRobot dataset.

    Handles:
    - Loading dataset from Hugging Face hub or local path
    - Building history and future frames (delta_timestamps)
    - Splitting into training and validation sets
    - Applying image and proprioceptive transformations
    - Creating optimized DataLoaders for training

    Args:
        repo_id: Hugging Face repo ID
        local_root: local dataset root directory
        visual_obs_names: names of the visual observations
        device: CPU or GPU
        batch_size: number of samples per batch
        shuffle: whether to shuffle the dataset
        train_split: training split ratio (0–1)
        num_workers: number of parallel CPU workers
        seed_val: base seed for reproducibility
        N_history: number of past frames (including current)
        N_chunk: number of future action frames
        fps: desired dataset frame rate to build delta_timestamps (Hz)
        print_ds_info: print dataset statistics

    Returns:
        (DataLoader, DataLoader): train and validation dataloaders
    """

    # Get dataset metadata
    dataset_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=local_root)

    # Build the delta_timestamps dict for LeRobotDataset history and future        
    delta_timestamps = build_delta_timestamps(dataset_meta.fps, fps_sampling, N_history, N_chunk, visual_obs_names)

    # Load the raw dataset (hub or local)
    dataset = LeRobotDataset(
        repo_id=repo_id,
        delta_timestamps=delta_timestamps,
        root=local_root,   
    )

    # print stats
    if print_ds_info:
        print(f"Number of episodes selected: {dataset.num_episodes}")
        print(f"Number of fps: {dataset.fps}")
        print(f"Camera keys: {dataset.meta.camera_keys}")
        print("Features:")
        pprint(dataset.features)
        features = dataset_to_policy_features(dataset.features)
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features}
        print("Output features: \n", output_features)
        print("Input features: \n", input_features)

    # fix seed for reproducibility
    if seed_val:
        worker_fn = make_worker_init_fn(seed_val)
    else:
        worker_fn = None   

    # Random split between training and validation
    num_items = len(dataset)
    print("Total number of frames in the dataset: ", num_items)
    num_train = int(num_items * train_split)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(dataset, [num_train, num_val], generator=torch.Generator())
    print("Total number of frames in the training dataset: ", num_train)
    print("Total number of frames in the validation dataset: : ", num_val)

    # Get transformation pipeline (augmentations, normalization, etc.)
    train_tf = CustomTransforms(visual_obs_names, train=True)
    val_tf = CustomTransforms(visual_obs_names, train=True)
    
    # Apply transforms
    transformed_train_ds = TransformedDataset(train_ds, train_tf)
    transformed_val_ds = TransformedDataset(val_ds, val_tf)

    # Wrap in DataLoaders

    train_loader = DataLoader(
        transformed_train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers, # number of CPU processes that load data in parallel
        pin_memory=device.type!="cpu", # allocate CPU memory as pinned for faster CPU to GPU transfers
        prefetch_factor=2, # each worker preloads 2 batches ahead
        persistent_workers=True, # keep workers alive between epochs 
        worker_init_fn=worker_fn, # ensures each worker has a reproducible deterministic seed
        drop_last=True # if the total number of samples is not divisible by batch_size, the last incomplete batch is dropped
    )

    val_loader = DataLoader(
        transformed_val_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type!="cpu",
        prefetch_factor=2,
        persistent_workers=True,
        worker_init_fn=worker_fn,
        drop_last=True
                        
    )

    return train_loader, val_loader



