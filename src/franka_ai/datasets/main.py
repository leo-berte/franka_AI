from franka_ai.datasets.load_dataset import make_dataloader

train_loader, train_dataset = make_dataloader(
    repo_id="lerobot/pusht",
    batch_size=32
)

make_dataloader(
    repo_id="my_user/softbag_pick_place",
    local_root="~/Documents/LeRobot_datasets",
    mode="local"
)