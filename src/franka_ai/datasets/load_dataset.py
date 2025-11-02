from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from franka_ai.datasets.transforms import get_transforms


# che differenza con mio codice wake_word che faccio train e test split?


def make_dataloader(
    repo_id="lerobot/pusht",
    batch_size=32,
    shuffle=True,
    num_workers=4,
    delta_timestamps=None,
    local_root=None,
):
    
    # 1. Load the raw dataset (hub or local)
    dataset = LeRobotDataset(
        repo_id=repo_id,
        delta_timestamps=delta_timestamps,
        root=local_root,
        mode="local" if local_root else "hub",
        # video_backend="pyav",   
    )

    # 2. Get transformation pipeline (augmentations, normalization, etc.)
    transform = get_transforms()
    dataset.set_transform(transform)

    # 3. Wrap in DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader, dataset
