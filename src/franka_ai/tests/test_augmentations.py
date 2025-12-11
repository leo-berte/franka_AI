import argparse
from pathlib import Path
import time
import torch
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

from franka_ai.dataset.transforms import CustomTransforms
from franka_ai.dataset.utils import get_configs_dataset
from franka_ai.models.utils import get_configs_models


"""
Run the code: 

python src/franka_ai/tests/test_augmentations.py --image test.jpeg --policy diffusion
"""


def parse_args():

    parser = argparse.ArgumentParser("Test dataset augmentations")

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="RGB image name"
    )

    parser.add_argument(
        "--policy",
        type=str,
        default="diffusion",
        choices=["diffusion", "act"],
        help="Policy name"
    )

    return parser.parse_args()

def benchmark_image_augmentations(
    tf,
    img,
    N_hist=8,
    n_iters=1000,
    device="cpu",
):
    
    """
    tf: CustomTransforms instance
    img: (1, C, H, W) tensor in [0,1]
    """

    assert img.dim() == 4 and img.shape[0] == 1

    # Move transforms to device
    tf.img_tf_train = tf.img_tf_train.to(device)

    # Replicate image → (N_hist, C, H, W)
    img_hist = img.repeat(N_hist, 1, 1, 1).to(device)

    # Warmup (important for GPU)
    with torch.no_grad():
        for _ in range(10):
            _ = tf.img_tf_train(img_hist)

    # Timing
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()

    with torch.no_grad():
        for _ in range(n_iters):
            _ = tf.img_tf_train(img_hist)

    torch.cuda.synchronize() if device == "cuda" else None
    t1 = time.perf_counter()

    total_time = t1 - t0
    avg_time = total_time / n_iters

    print(f"\n=== BENCHMARK ({device.upper()}) ===")
    print(f"N_hist       : {N_hist}")
    print(f"Iterations   : {n_iters}")
    print(f"Total time   : {total_time:.4f} s")
    print(f"Time / iter  : {avg_time*1e3:.3f} ms")


def main():

    args = parse_args()

    image_path = Path(f"images/{args.image}")
    out_dir = Path("images")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    dataloader_cfg, dataset_cfg, transforms_cfg = get_configs_dataset("configs/config/dataset.yaml")
    models_cfg = get_configs_models("configs/config/models.yaml")

    # Init transforms (training mode to enable augmentations)
    tf = CustomTransforms(
        dataset_cfg=dataset_cfg,
        transforms_cfg=transforms_cfg,
        model_cfg=models_cfg[args.policy],
        train=True,
    )

    # -------------------
    # IMAGE TEST
    # -------------------

    img = Image.open(image_path).convert("RGB")
    img = TF.to_tensor(img)          # (C,H,W), [0,1]
    img = img.unsqueeze(0).float()   # (1,C,H,W)

    print("Image input:")
    print("  shape:", img.shape)
    print("  range:", img.min().item(), img.max().item())

    save_image(img, out_dir / "image_before.png")

    img_tf = tf.img_tf_train(img)

    print("Image after transforms:")
    print("  shape:", img_tf.shape)
    print("  range:", img_tf.min().item(), img_tf.max().item())

    save_image(img_tf, out_dir / "image_after.png")


    # -------------------
    # JOINT VELOCITY TEST
    # -------------------

    N_history = tf.N_history
    D_vel = 7

    joint_vel = torch.zeros(1, N_history, D_vel)

    print("\nJoint velocity BEFORE:")
    print(joint_vel)

    joint_vel_tf = tf.joint_vel_transforms(joint_vel)

    print("\nJoint velocity AFTER:")
    print(joint_vel_tf)

    print("Δ mean:", (joint_vel_tf - joint_vel).mean().item())
    print("Δ std :", (joint_vel_tf - joint_vel).std().item())


    # -------------------
    # JOINT TORQUE TEST
    # -------------------

    D_tau = 7

    joint_tau = torch.zeros(1, N_history, D_tau)

    print("\nJoint torque BEFORE:")
    print(joint_tau)

    joint_tau_tf = tf.joint_torque_transforms(joint_tau)

    print("\nJoint torque AFTER:")
    print(joint_tau_tf)

    print("Δ mean:", (joint_tau_tf - joint_tau).mean().item())
    print("Δ std :", (joint_tau_tf - joint_tau).std().item())

    # ---------------------------
    # BENCHMARK CPU Vs CPU TIMING
    # ---------------------------

    print(f"\nStarting Benchmark analysis...")

    benchmark_image_augmentations(
        tf=tf,
        img=img,
        N_hist=tf.N_history,
        n_iters=1000,
        device="cuda",
    )

    benchmark_image_augmentations(
        tf=tf,
        img=img,
        N_hist=tf.N_history,
        n_iters=1000,
        device="cpu",
    )


if __name__ == "__main__":
    
    main()