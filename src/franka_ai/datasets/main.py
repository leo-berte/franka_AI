from franka_ai.datasets.load_dataset import make_dataloader


# load dataset from HF
train_loader, val_loader = make_dataloader(
    repo_id="lerobot/pusht",
    batch_size=32,
    N_history=10, 
    N_chunk=5,
    fps=10,
    print_ds_info=True
)

# load dataset from local folder

"""
~/Documents/LeRobot_datasets/
└── my_user/
    └── softbag_pick_place/
        ├── metadata/
        ├── videos/
        ├── data/
        └── version.txt
"""

# train_loader, val_loader = make_dataloader(
#     repo_id="my_user/softbag_pick_place",
#     local_root="~/Documents/LeRobot_datasets",
#     batch_size=32,
#     print_ds_info=True
# )

# iterate over dataloader
for batch in train_loader:
    print(f"{batch['observation.image'].shape=}")  # (32, N_h, c, h, w)
    print(f"{batch['observation.state'].shape=}")  # (32, N_h, 14)
    print(f"{batch['action'].shape=}")  # (32, N_c, 14)
    break

