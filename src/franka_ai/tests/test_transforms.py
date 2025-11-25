import torch

from franka_ai.dataset.transforms import CustomTransforms


"""
Run the code: python src/franka_ai/tests/test_transforms.py
"""


def main():

    print("\nTEST 1 — Identity Quaternion")
    q = torch.tensor([0.,0.,0.,1.])
    aa1 = CustomTransforms.quaternion2axis_angle(q)
    aa2 = CustomTransforms.quaternion2axis_angle2(q)
    print("AA scipy-like:", aa1)
    print("AA torch:", aa2)

    print("\nTEST 1b — Identity Axis-Angle")
    aa = torch.tensor([0.,0.,0.])
    q1 = CustomTransforms.axis_angle2quaternion(aa)
    q2 = CustomTransforms.axis_angle2quaternion2(aa)
    print("Quat scipy-like:", q1)
    print("Quat torch:", q2)

    print("\nTEST 2 — 180 degrees around X")
    angle = torch.tensor(torch.pi)
    aa = torch.tensor([angle, 0., 0.])
    q_scipy = CustomTransforms.axis_angle2quaternion(aa)
    q_torch = CustomTransforms.axis_angle2quaternion2(aa)
    print("Quat scipy-like:", q_scipy)
    print("Quat torch:", q_torch)
    aa_scipy = CustomTransforms.quaternion2axis_angle(q_scipy)
    aa_torch  = CustomTransforms.quaternion2axis_angle2(q_scipy)
    print("AA scipy-like (roundtrip):", aa_scipy)
    print("AA torch (roundtrip):", aa_torch)

    print("\nTEST 3 — Random orientation stress test with history")
    q_rand = torch.randn(5, 4)
    q_rand = q_rand / q_rand.norm(dim=-1, keepdim=True)
    aa_scipy = CustomTransforms.quaternion2axis_angle(q_rand)
    aa_torch = CustomTransforms.quaternion2axis_angle2(q_rand)
    print("AA scipy-like:", aa_scipy)
    print("AA torch:", aa_torch)
    q_scipy_back = CustomTransforms.axis_angle2quaternion(aa_scipy)
    q_torch_back = CustomTransforms.axis_angle2quaternion2(aa_torch)
    print("Recovered quats scipy:", q_scipy_back)
    print("Recovered quats torch:", q_torch_back)


if __name__ == "__main__":
    main()