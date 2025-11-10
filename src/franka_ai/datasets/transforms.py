from scipy.spatial.transform import Rotation as R
import kornia.augmentation as K
import torchvision.transforms as T
import torch


# TODO: 
# 1) Check values resize, bright, ..
# 2) normalization of features HERE?
# 3) gripper continuous2dicsrete conversion
# 4) orientation conversions

# -- panda_gripper/gripper_command: it's our action, an it will be a discrete value (0.0 or 1.0) --> check type
# -- panda_gripper/width: it's our observation, an it will be a continuous value between 0.0 and 1.0 --> check type

class CustomTransforms():

    """
    Build and apply custom transformations to dataset samples.

    Includes:
    - Image preprocessing and augmentations (resize, jitter, affine, blur)
    - Gaussian noise injection to proprioceptive data (joint positions, velocities, torques)
    The same transformation or noise injection is applied for every data in the history to mantain temporal correlation.

    Args:
        train: boolean to decide whether to apply training augmentations or inference preprocessing.
        visual_obs_names: names of the visual observations to which I want to apply transformations.
    """

    def __init__(self, visual_obs_names, train=False):
        
        self.train = train
        self.visual_obs_names = visual_obs_names

        self.joint_pos_std_dev = 0.002 # σ ≈ 0.002–0.01 rad (≈ 0.1°–0.6°)
        self.joint_vel_std_dev = 0.002 # σ ≈ 0.02 rad/s
        self.joint_torque_std_dev = 0.002 # σ ≈ 0.02 rad/s 
        self.cart_position_std_dev = 0.002 # σ ≈ 0.5–2 mm
        self.cart_orientation_std_dev = 0.002 # (axis-angle): σ ≈ 0.5–2° 
        
        # convert img to [-1,1] (both training and inference)
        base_tf = torch.nn.Sequential(
            K.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]),
                        std=torch.tensor([0.5, 0.5, 0.5]))
        )

        # training augmentation
        train_tf = torch.nn.Sequential(
                K.Resize((224, 224)),
                K.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05,
                              p=1.0, same_on_batch=True),
                K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.3, same_on_batch=True),
                K.RandomAffine(degrees=5, translate=(0.05, 0.05),
                               p=0.5, same_on_batch=True))

        self.img_tf_train = torch.nn.Sequential(train_tf, base_tf)
        self.img_tf_inference = base_tf

    def joint_pos_transforms(self, v):
        noise = torch.randn(1, v.shape[-1]) * self.joint_pos_std_dev # v: (N_h, D) (history) → noise: (1, D)
        return v + noise
    
    def joint_vel_transforms(self, v):
        noise = torch.randn(1, v.shape[-1]) * self.joint_vel_std_dev
        return v + noise
    
    def joint_torque_transforms(self, v):
        noise = torch.randn(1, v.shape[-1]) * self.joint_torque_std_dev
        return v + noise
    
    def curr_cart_position_transforms(self, v):
        pass
    
    def curr_cart_orientation_transforms(self, v):
        pass

    def gripper_continuous2discrete(self, value, threshold=0.5):
        return (value > threshold).float() # returns 0 (closed) or 1 (open)

    def quaternion2axis_angle(self, q):
    
        """
        Convert quaternion(s) to axis-angle representation using SciPy.
        q: (..., 4) tensor with format (w, x, y, z)
        returns: (..., 3) tensor with axis-angle representation
        """

        q_np = q.detach().cpu().numpy()
        # reorder from (w, x, y, z) → (x, y, z, w) as expected by SciPy
        q_np = q_np[..., [1, 2, 3, 0]]

        r = R.from_quat(q_np)
        aa = r.as_rotvec()  # returns axis * angle, shape (..., 3)
        return torch.from_numpy(aa).to(q.device, dtype=q.dtype)

    def axis_angle2quaternion(self, axis_angle):
        
        """
        Convert axis-angle representation(s) to quaternion using SciPy.
        axis_angle: (..., 3) tensor representing axis-angle
        returns: (..., 4) tensor with format (w, x, y, z)
        """

        aa_np = axis_angle.detach().cpu().numpy()
        r = R.from_rotvec(aa_np)
        q_np = r.as_quat()  # returns (x, y, z, w)
        q_np = q_np[..., [3, 0, 1, 2]]  # reorder to (w, x, y, z)
        return torch.from_numpy(q_np).to(axis_angle.device, dtype=axis_angle.dtype)

    def transform(self, sample):
        
        for k, v in sample.items(): # each sample contains the N_h dimension (but no B dimension)

            # Images
            if self.visual_obs_names and k == self.visual_obs_names:
                    v = v.to(torch.float32) # convert from uint8 [0,255] to tensor
                    v = v / 255.0 if v.max() > 1.5 else v # convert to [0,1]
                    sample[k] = self.img_tf_train(v) if self.train else self.img_tf_inference(v)

            # # Joint positions --> però poi devo cambiare anche Tcurr con DKINE ??
            # elif "observation.state" == k:
            #         sample[k] = self.joint_pos_transforms(v) if self.train else v

            # # Joint velocities
            # elif "observation.joint_velocities" == k:
            #         sample[k] = self.joint_vel_transforms(v) if self.train else v

            # # Joint torques
            # elif "observation.joint_torques" == k:
            #         sample[k] = self.joint_torque_transforms(v) if self.train else v

        return sample