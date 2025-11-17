from torch.utils.data import Dataset, DataLoader, random_split
from pprint import pprint
import torch
import random

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from franka_ai.dataset.utils import build_delta_timestamps, print_dataset_info
from franka_ai.utils.seed_everything import make_worker_init_fn


class TransformedDataset(Dataset):
    
    """
    Wrap a base dataset and apply custom transformations to each sample.

    Args:
        base_dataset: original dataset
        custom_transforms: transformation pipeline applied to each sample
    """

    def __init__(self, base_dataset, custom_transforms):
        
        # init
        self.base = base_dataset
        self.custom_transforms = custom_transforms

        # extract dataset features to be removed
        self.skip_features = []
        for v in self.custom_transforms.feature_groups["REMOVE"]:
            self.skip_features.append(v)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):

        # get sample
        sample = self.base[idx]

        # apply transforms
        sample = self.custom_transforms.transform(sample) 

        # drop unwanted features
        sample = {k: v for k, v in sample.items() if k not in self.skip_features}

        return sample


def make_dataloader(
    repo_id=None, 
    local_root=None, 
    dataloader_cfg=None,
    feature_groups=None,
    transforms_train=None,
    transforms_val=None,
    transforms_stats=None,
    **overrides
):
    
    """
    
    TO UPDATE
    
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

    # Dataloader params (batch_size, seed_val, ..) passed via argument will override yaml
    dl_cfg = {**(dataloader_cfg or {}), **overrides}

    # Extract parameters
    N_history          = dl_cfg["N_history"]
    N_chunk            = dl_cfg["N_chunk"]
    batch_size         = dl_cfg["batch_size"]
    shuffle            = dl_cfg["shuffle"]
    train_split        = dl_cfg["train_split"]
    fps_sampling_hist  = dl_cfg["fps_sampling_hist"]
    fps_sampling_chunk = dl_cfg["fps_sampling_chunk"]
    num_workers        = dl_cfg["num_workers"]
    prefetch_factor    = dl_cfg["prefetch_factor"]
    seed_val           = dl_cfg["seed_val"]
    print_ds_info      = dl_cfg["print_ds_info"]
    device             = torch.device(dl_cfg["device"])

    # Get dataset metadata
    dataset_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=local_root)

    # Build the delta_timestamps dict for LeRobotDataset history and future        
    delta_timestamps = build_delta_timestamps(feature_groups, N_history, N_chunk, dataset_meta.fps, fps_sampling_hist, fps_sampling_chunk)

    # Generate random list of indeces covering all the episodes
    num_episodes = dataset_meta.total_episodes
    episode_ids = list(range(num_episodes))
    random.shuffle(episode_ids)
    
    # Split indeces for training and validation
    num_train_episodes = int(num_episodes * train_split)
    train_episodes = episode_ids[:num_train_episodes]
    val_episodes = episode_ids[num_train_episodes:]
    
    # Load the raw dataset (hub or local)
    train_ds = LeRobotDataset(
        repo_id=repo_id,
        root=local_root,   
        delta_timestamps=delta_timestamps,
        episodes=[0] # train_episodes
    )

    # Load the raw dataset (hub or local)
    val_ds = LeRobotDataset(
        repo_id=repo_id,
        root=local_root,   
        delta_timestamps=delta_timestamps,
        episodes=[0] # val_episodes
    )

    # print stats
    if print_ds_info:
        print_dataset_info(train_ds, "training dataset")
        print_dataset_info(val_ds, "validation dataset")

    # fix seed for reproducibility
    if seed_val is not None and seed_val >= 0:
        worker_fn = make_worker_init_fn(seed_val)
    else:
        worker_fn = None   

    # Apply transforms
    transformed_train_ds = TransformedDataset(train_ds, transforms_train)
    transformed_val_ds = TransformedDataset(val_ds, transforms_val)
    transformed_stats_ds = TransformedDataset(train_ds, transforms_stats)

    # Wrap in DataLoaders

    train_loader = DataLoader(
        transformed_train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers, # number of CPU processes that load data in parallel
        pin_memory=device.type!="cpu", # allocate CPU memory as pinned for faster CPU to GPU transfers
        prefetch_factor=prefetch_factor, # each worker preloads 2 batches ahead
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
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        worker_init_fn=worker_fn,
        drop_last=True            
    )

    stats_loader = DataLoader(
        transformed_stats_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type!="cpu",
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        worker_init_fn=worker_fn,
        drop_last=True    
    )

    return train_loader, val_loader, stats_loader



