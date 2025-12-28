from scipy.spatial.transform import Rotation as R
import torch
import numpy as np

from franka_ai.dataset.transforms import CustomTransforms


"""
Run the code: python src/franka_ai/tests/test_orientations.py
"""

def axis_angle_error(aa_pred, aa_gt):

    """
    aa_pred, aa_gt: (..., 3) axis-angle (axis * angle) in rads
    """

    q_pred = CustomTransforms.axis_angle2quaternion(aa_pred).numpy()
    q_gt   = CustomTransforms.axis_angle2quaternion(aa_gt).numpy()

    return quat_angle_error(q_pred, q_gt)

def quat_angle_error(q1, q2):

    """
    q1, q2: (..., 4) quaternion [x,y,z,w]
    """

    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)

    dot = np.sum(q1 * q2, axis=-1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)

    return 2 * np.arccos(dot)

def quaternion2axis_angle2(q):

    """
    Convert quaternion(s) to axis-angle representation using SciPy.
    q: (..., 4) tensor with format (x, y, z, w)
    returns: (..., 3) tensor with axis-angle representation
    """

    q_np = q.numpy()
    r = R.from_quat(q_np)
    aa = r.as_rotvec()  # returns axis * angle, shape (..., 3)   
    return torch.from_numpy(aa).to(q.device, dtype=q.dtype)

def axis_angle2quaternion2(axis_angle):
    
    """
    Convert axis-angle representation(s) to quaternion using SciPy.
    axis_angle: (..., 3) tensor representing axis-angle
    returns: (..., 4) tensor with format (w, x, y, z)
    """

    aa_np = axis_angle.numpy()
    r = R.from_rotvec(aa_np)
    q_np = r.as_quat()  # returns (x, y, z, w)
    return torch.from_numpy(q_np).to(axis_angle.device, dtype=axis_angle.dtype)
    


def main():

    print("\nTEST 1 — Identity Quaternion")
    q = torch.tensor([0.,0.,0.,1.])
    aa1 = CustomTransforms.quaternion2axis_angle(q)
    aa2 = quaternion2axis_angle2(q)
    print("AA torch:", aa1)
    print("AA scipy-like:", aa2)

    print("\nTEST 1b — Identity Axis-Angle")
    aa = torch.tensor([0.,0.,0.])
    q1 = CustomTransforms.axis_angle2quaternion(aa)
    q2 = axis_angle2quaternion2(aa)
    print("Quat torch:", q1)
    print("Quat scipy-like:", q2)

    print("\nTEST 2 — 180 degrees around X")
    angle = torch.tensor(torch.pi)
    aa = torch.tensor([angle, 0., 0.])
    q_scipy = axis_angle2quaternion2(aa)
    q_torch = CustomTransforms.axis_angle2quaternion(aa)
    print("Quat scipy-like:", q_scipy)
    print("Quat torch:", q_torch)
    aa_scipy = quaternion2axis_angle2(q_scipy)
    aa_torch  = CustomTransforms.quaternion2axis_angle(q_scipy)
    print("AA scipy-like (roundtrip):", aa_scipy)
    print("AA torch (roundtrip):", aa_torch)

    print("\nTEST 3 — Random orientation stress test with history")
    q_rand = torch.randn(5, 4)
    q_rand = q_rand / q_rand.norm(dim=-1, keepdim=True)
    print("Q start:", q_rand)
    aa_scipy = quaternion2axis_angle2(q_rand)
    aa_torch = CustomTransforms.quaternion2axis_angle(q_rand)
    print("AA scipy-like:", aa_scipy)
    print("AA torch:", aa_torch)
    q_scipy_back = axis_angle2quaternion2(aa_scipy)
    q_torch_back = CustomTransforms.axis_angle2quaternion(aa_torch)
    print("Recovered quats scipy:", q_scipy_back)
    print("Recovered quats torch:", q_torch_back)

    print("\nTEST 4 — Debug specific transforms")
    
    aa = torch.tensor([-2.9598, -1.5133,  0.0645])
    print("AA start:", aa)
    q_scipy = axis_angle2quaternion2(aa)
    q_torch = CustomTransforms.axis_angle2quaternion(aa)
    print("Q scipy-like:", q_scipy)
    print("Q torch:", q_torch)

    aa_scipy = quaternion2axis_angle2(q_scipy)
    aa_torch  = CustomTransforms.quaternion2axis_angle(q_torch)
    print("AA scipy-like (roundtrip):", aa_scipy)
    print("AA torch (roundtrip):", aa_torch)

    # compare
    aa_mod = torch.linalg.norm(aa)
    print(np.rad2deg(aa_mod), aa/aa_mod)
    aa_mod_roundtrip = torch.linalg.norm(aa_torch)
    print(np.rad2deg(aa_mod_roundtrip), aa_scipy/aa_mod_roundtrip)
    aa_error_rad = axis_angle_error(aa, aa_torch)
    aa_error_deg = np.rad2deg(aa_error_rad)
    print(aa_error_deg)

    q = torch.tensor([-0.8878, -0.4537,  0.0193, -0.0750])
    print("Q start:", q)
    aa_scipy = quaternion2axis_angle2(q)
    aa_torch = CustomTransforms.quaternion2axis_angle(q)
    print("AA scipy-like:", aa_scipy)
    print("AA torch:", aa_torch)

    q_scipy = axis_angle2quaternion2(aa_scipy)
    q_torch  = CustomTransforms.axis_angle2quaternion(aa_torch)
    print("Q scipy-like (roundtrip):", q_scipy)
    print("Q torch (roundtrip):", q_torch)

    # compare
    quat_error_rad = quat_angle_error(q.numpy(), q_torch.numpy())
    quat_error_deg = np.degrees(quat_error_rad)
    print(quat_error_deg)



if __name__ == "__main__":
    main()