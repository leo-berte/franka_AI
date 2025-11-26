import kornia.augmentation as K
import torchvision.transforms as T
import torch
import numpy as np



# TODO: 

# 2) check rviz depth



# 1) add relative vs absolute cart pose as actions/state

# 1) Check values resize, bright, ..
# 6) bring kornia on GPU? Ma ha senso spostare i dati già su GPU qui? non fare lavoro doppio
# 9) salva delle immagini post trasformazione nel training e vedi che forma hanno
# 10) immagini post trasformazioni sono ancora in 0,1? xke datasets stats si aspetta quello




class CustomTransforms():

    """

    TO UPDATE

    Build and apply custom transformations to dataset samples.

    Includes:
    - Image preprocessing and augmentations (resize, jitter, affine, blur)
    - Gaussian noise injection to proprioceptive data (joint positions, velocities, torques)
    The same transformation or noise injection is applied for every data in the history to mantain temporal correlation.

    Args:
        train: boolean to decide whether to apply training augmentations or inference preprocessing.
        feature_groups: eature names used in the dataset.
    """

    def __init__(self, dataloader_cfg, dataset_cfg, transforms_cfg, train=False):
        
        self.train = train
        self.feature_groups = dataset_cfg["features"]
        state_ranges = dataset_cfg["state_slices"]
        self.state_slices = {k: slice(v[0], v[1]) for k, v in state_ranges.items()}

        # history and chunk sizes
        self.N_history = dataloader_cfg["N_history"]
        self.N_chunk = dataloader_cfg["N_chunk"]

        # image resize
        self.img_resize = transforms_cfg["img_resize"]

        # noise stds
        noise = transforms_cfg["noise_std_dev"]
        self.joint_pos_std_dev = noise["joint_pos"]
        self.joint_vel_std_dev = noise["joint_vel"]
        self.joint_torque_std_dev = noise["joint_torque"]
        self.cart_position_std_dev = noise["cart_position"]
        self.cart_orientation_std_dev = noise["cart_orientation"]

        # image augmentations
        aug_cfg = transforms_cfg["augmentations"]
        
        # convert img to [0,1] + resize (both training and inference)
        base_tf_pre = torch.nn.Sequential(
            K.Resize(tuple(self.img_resize)),
            # K.Normalize(mean=torch.tensor([0.0, 0.0, 0.0]),
            #             std=torch.tensor([255.0, 255.0, 255.0]))
        )

        # training image augmentations
        train_tf = torch.nn.Sequential(
            K.ColorJitter(
                brightness=aug_cfg["color_jitter"]["brightness"],
                contrast=aug_cfg["color_jitter"]["contrast"],
                saturation=aug_cfg["color_jitter"]["saturation"],
                hue=aug_cfg["color_jitter"]["hue"],
                p=aug_cfg["color_jitter"]["p"],
                same_on_batch=True,
                ),
            K.RandomGaussianBlur(
                kernel_size=tuple(aug_cfg["gaussian_blur"]["kernel_size"]),
                sigma=tuple(aug_cfg["gaussian_blur"]["sigma"]),
                p=aug_cfg["gaussian_blur"]["p"],
                same_on_batch=True,
                ),
            K.RandomAffine(
                degrees=aug_cfg["random_affine"]["degrees"],
                translate=tuple(aug_cfg["random_affine"]["translate"]),
                p=aug_cfg["random_affine"]["p"],
                same_on_batch=True,
                )
        )

        # define full pipeline for both training and inference
        self.img_tf_inference = torch.nn.Sequential(base_tf_pre)
        self.img_tf_train = torch.nn.Sequential(base_tf_pre, train_tf)
    
    def joint_pos_transforms(self, v):
        noise = torch.randn_like(v[:1,:]) * self.joint_pos_std_dev # v: (N_h, D) (history) → noise: (1, D)
        return v + noise
    
    def joint_vel_transforms(self, v):
        noise = torch.randn_like(v[:1,:]) * self.joint_vel_std_dev
        return v + noise
    
    def joint_torque_transforms(self, v):
        noise = torch.randn_like(v[:1,:]) * self.joint_torque_std_dev
        return v + noise
    
    def curr_cart_position_transforms(self, v):
        pass
    
    def curr_cart_orientation_transforms(self, v):
        pass

    def gripper_continuous2discrete(self, value, gripper_half_width=0.037):
        return (value > gripper_half_width).float() # returns 0 (closed) or 1 (open)

    @staticmethod
    def quaternion2axis_angle(q):

        """
        q: (..., 4)  in (x, y, z, w)
        returns: (..., 3) axis-angle
        """

        # Ensure normalized quaternion
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)

        x, y, z, w = q.unbind(-1)

        angle = 2 * torch.atan2(torch.sqrt(x*x + y*y + z*z), w)
        axis = torch.stack([x, y, z], dim=-1)
        axis_norm = axis.norm(dim=-1, keepdim=True) + 1e-8
        axis = axis / axis_norm

        # Fix sign ambiguity: enforce SciPy-like convention: angle ∈ [-π, π]
        angle = ((angle + torch.pi) % (2 * torch.pi)) - torch.pi

        return axis * angle.unsqueeze(-1)
    
    @staticmethod
    def axis_angle2quaternion(aa):

        """
        aa: (..., 3) axis * angle
        returns (..., 4) quaternion (x, y, z, w)
        """

        angle = torch.norm(aa, dim=-1, keepdim=True) + 1e-8
        axis = aa / angle

        half = angle * 0.5
        w = torch.cos(half)
        xyz = axis * torch.sin(half)

        return torch.cat([xyz, w], dim=-1)

    def transform(self, sample):
        
        for k, v in sample.items(): # each sample contains the N_h dimension (but no B dimension)
            
            # images
            if k in self.feature_groups["VISUAL"]:

                v = v.to(torch.float32) # convert data to tensor float32
                sample[k] = self.img_tf_train(v) if self.train else self.img_tf_inference(v)

            # state
            if k in self.feature_groups["STATE"]:

                v = v.to(torch.float32) # convert data to tensor float32

                # add noise on joint velocities
                joint_vel = v[..., self.state_slices["qdot"]]
                joint_vel_aug = self.joint_vel_transforms(joint_vel) if self.train else joint_vel

                # add noise on joint torques
                joint_torque = v[..., self.state_slices["tau"]]
                joint_torque_aug = self.joint_torque_transforms(joint_torque) if self.train else joint_torque

                # convert to discrete gripper state (0.0 or 1.0)
                gripper_cont = v[..., self.state_slices["gripper"]]
                gripper_disc = self.gripper_continuous2discrete(gripper_cont)
                
                # convert orientation to axis-angle
                q_orientation = v[..., self.state_slices["ee_quaternion"]]
                aa_orientation = self.quaternion2axis_angle(q_orientation)

                # rebuild state vector
                v_new = torch.cat([
                    v[..., self.state_slices["q"]],
                    joint_vel_aug,
                    joint_torque_aug,
                    v[..., self.state_slices["ee_pos"]],
                    aa_orientation,
                    gripper_disc,
                    v[..., self.state_slices["ext_force"]]                    
                ], dim=-1)

                sample[k] = v_new

            # action
            if k in self.feature_groups["ACTION"]:

                v = v.to(torch.float32) # convert data to tensor float32
                past_actions = v[:self.N_history, :]
                future_actions = v[self.N_history:, :]
                sample[k] = future_actions

        # append past actions to state
        state_ft_name = self.feature_groups["STATE"][0]
        sample[state_ft_name] = torch.cat([
            sample[state_ft_name],
            past_actions,
        ], dim=-1)

        return sample