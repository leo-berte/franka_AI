import torch
import numpy as np

from franka_ai.utils.robotics_math import *

"""
Run the code: python src/franka_ai/tests/test_robotics_math.py
"""


torch.set_printoptions(precision=6, sci_mode=False)


def main():

    device = "cpu"
    dtype = torch.float64
    N = 1

    print("=== Testing PyTorch3D rotation conversions ===\n")

    # --------------------------------------------------
    # Generazione rotazioni di test
    # --------------------------------------------------
    quats = random_quaternions(N, dtype=dtype, device=device)
    mats = quaternion_to_matrix(quats)
    axis_angles = quaternion_to_axis_angle(quats)

    print(f"Batch size: {N}")
    print(f"Dtype: {dtype}")
    print()

    # --------------------------------------------------
    # quaternion <-> matrix
    # --------------------------------------------------
    print("---- quaternion <-> matrix ----")

    mats_from_q = quaternion_to_matrix(quats)
    quats_back = matrix_to_quaternion(mats_from_q)
    print("quats: ", quats)
    print("quats_back: ", quats_back)

    err_q1 = vector_error(quats, quats_back)
    err_q2 = quaternion_geodesic_error(quats, quats_back)
    print(f"Quaternion round-trip error1: {err_q1:.3e}")
    print("Quaternion round-trip error2: ", err_q2)

    # --------------------------------------------------
    # axis_angle <-> matrix (fast=True)
    # --------------------------------------------------
    print("\n---- axis_angle <-> matrix (fast=True) ----")

    mats_from_aa = axis_angle_to_matrix(axis_angles, fast=True)
    aa_back = matrix_to_axis_angle(mats_from_aa, fast=True)
    print("axis_angles: ", axis_angles)
    print("aa_back: ", aa_back)

    err_aa1 = vector_error(axis_angles, aa_back)
    err_aa2 = axis_angle_geodesic_error(axis_angles, aa_back)
    print(f"Axis-angle round-trip error1: {err_aa1:.3e}")
    print("Axis-angle round-trip error2: ", err_aa2)

    # --------------------------------------------------
    # axis_angle <-> quaternion
    # --------------------------------------------------
    print("\n---- axis_angle <-> quaternion ----")

    quats_from_aa = axis_angle_to_quaternion(axis_angles) # standardize_quaternion() NOT USED
    aa_back2 = quaternion_to_axis_angle(quats_from_aa)
    print("axis_angles: ", axis_angles)
    print("aa_back2: ", aa_back2)

    err_aa_q1 = vector_error(axis_angles, aa_back2)
    err_aa_q2 = axis_angle_geodesic_error(axis_angles, aa_back2)
    print(f"Axis-angle <-> quaternion error1: {err_aa_q1:.3e}")
    print("Axis-angle <-> quaternion error2: ", err_aa_q2)

    # --------------------------------------------------
    # matrix <-> quaternion
    # --------------------------------------------------
    print("\n---- matrix <-> quaternion ----")

    mats_back = quaternion_to_matrix(matrix_to_quaternion(mats))
    err_m = rotation_matrix_geodesic_error(mats, mats_back)
    print("mats: ", mats)
    print("mats_back: ", mats_back)

    print("Matrix round-trip error: ", err_m)

    # --------------------------------------------------
    # Test Identity
    # --------------------------------------------------
    print("\n---- Identity Quaternion ----")

    q = torch.tensor([1.,0.,0.,0.])
    aa = quaternion_to_axis_angle(q)
    print("Q:", q)
    print("AA:", aa)

    print("\n---- Identity Axis-Angle ----")

    aa = torch.tensor([0.,0.,0.])
    q = axis_angle_to_quaternion(aa)
    print("AA:", aa)
    print("Quat:", q)

    # --------------------------------------------------
    # Test Rotation 180 degrees around X
    # --------------------------------------------------
    print("\n---- 180 degrees around X ----")

    angle = torch.tensor(torch.pi)
    aa = torch.tensor([angle, 0., 0.])
    print("AA:", aa)
    q = axis_angle_to_quaternion(aa)
    print("Quat:", q)
    aa = quaternion_to_axis_angle(q)
    print("AA (roundtrip):", aa)

    # --------------------------------------------------
    # Test edge case
    # --------------------------------------------------

    print("\n---- edge case axis-angle round-trip ----")

    aa = torch.tensor([-2.9598, -1.5133,  0.0645])
    print("AA start:", aa)
    q_meta = axis_angle_to_quaternion(aa) # standardize_quaternion() NOT USED
    print("Q meta:", q_meta)
    aa_meta  = quaternion_to_axis_angle(q_meta)
    print("AA meta (roundtrip):", aa_meta)

    # compare
    aa_mod = torch.linalg.norm(aa)
    print(np.rad2deg(aa_mod), aa/aa_mod)
    aa_mod_roundtrip = torch.linalg.norm(aa_meta)
    print(np.rad2deg(aa_mod_roundtrip), aa_meta/aa_mod_roundtrip)
    aa_error_rad = axis_angle_geodesic_error(aa, aa_meta)
    aa_error_deg = np.rad2deg(aa_error_rad)
    print(aa_error_deg)

    print("\n---- edge case q Vs -q ----")

    q_start_meta1 = torch.tensor([0.550899, -0.535585,  0.162067,  0.619188])
    q_start_meta1 = standardize_quaternion(q_start_meta1)
    q_start_meta2 = torch.tensor([-0.550899, 0.535585,  -0.162067,  -0.619188])
    q_start_meta2 = standardize_quaternion(q_start_meta2)
    print("q_start_meta1:", q_start_meta1)
    aa_meta1 = quaternion_to_axis_angle(q_start_meta1)
    print("AA meta1:", aa_meta1)
    q_meta1 = axis_angle_to_quaternion(aa_meta1) # standardize_quaternion() NOT USED
    print("q_meta1 (roundtrip):", q_meta1)

    # compare
    quat_error_rad = quaternion_geodesic_error(q_start_meta1, q_meta1)
    quat_error_deg = np.degrees(quat_error_rad)
    print(quat_error_deg)

    # --------------------------------------------------
    # Test matrix <-> 6D
    # --------------------------------------------------
    print("\n---- matrix <-> 6D ----")

    mats_from_6d = rotation_6d_to_matrix(matrix_to_rotation_6d(mats))
    rot6d_back = matrix_to_rotation_6d(mats_from_6d)

    print("Original matrices:\n", mats)
    print("Matrices from 6D:\n", mats_from_6d)
    print("6D back from matrices:\n", rot6d_back)

    # Errore Frobenius sulle matrici
    err_mat6d = rotation_matrix_geodesic_error(mats, mats_from_6d)
    print("Matrix <-> 6D round-trip error:", err_mat6d)

    # Errore 6D
    err_6d = vector_error(matrix_to_rotation_6d(mats), rot6d_back)
    print("6D vector round-trip error:", err_6d)

    # --------------------------------------------------
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()