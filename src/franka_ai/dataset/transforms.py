import kornia.augmentation as K
import numpy as np
import torch

from franka_ai.utils.robotics_math import *


# TODO: 

# per come è ora, se uso in actions ee_rel ma non nello state, si rompe
# update description of methods/classes in the repo



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

    def __init__(self, dataset_cfg, transforms_cfg, model_cfg, train=False):
        
        self.train = train

        # dataset state and action indeces
        self.feature_groups = dataset_cfg["features"]
        state_ranges = dataset_cfg["state_slices"]
        self.state_slices = {k: slice(v[0], v[1]) for k, v in state_ranges.items()}
        action_ranges = dataset_cfg["action_slices"]
        self.action_slices = {k: slice(v[0], v[1]) for k, v in action_ranges.items()}

        # history and chunk sizes
        self.N_history = model_cfg["params"].get("n_obs_steps") or model_cfg["params"].get("N_history")
        self.N_chunk = model_cfg["params"].get("horizon") or model_cfg["params"].get("chunk_size") or model_cfg["params"].get("N_chunk")

        # state augmentations
        state_aug_cfg = transforms_cfg["state"]["augmentations"]
        self.joint_vel_std_dev = state_aug_cfg["noise_std_dev"]["joint_vel"]
        self.joint_torque_std_dev = state_aug_cfg["noise_std_dev"]["joint_torque"]

        # orientations
        self.orientation_type = transforms_cfg["orientations"]["type"]
        
        # state transforms
        self.use_past_actions = transforms_cfg["state"]["use_past_actions"]
        self.include_states = transforms_cfg["state"]["include"]

        # action transforms
        self.include_actions = transforms_cfg["action"]["include"]

        # image resize
        self.img_resize = transforms_cfg["visual"]["img_resize"]

        # image augmentations
        vis_aug_cfg = transforms_cfg["visual"]["augmentations"]
        
        # convert img to [0,1] + resize (both training and inference)
        base_tf_pre = torch.nn.Sequential(
            K.Resize(tuple(self.img_resize)),
            # K.Normalize(mean=torch.tensor([0.0, 0.0, 0.0]),
            #             std=torch.tensor([255.0, 255.0, 255.0]))
        )

        # training image augmentations
        train_tf = torch.nn.Sequential(
            K.ColorJitter(
                brightness=vis_aug_cfg["color_jitter"]["brightness"],
                contrast=vis_aug_cfg["color_jitter"]["contrast"],
                saturation=vis_aug_cfg["color_jitter"]["saturation"],
                hue=vis_aug_cfg["color_jitter"]["hue"],
                p=vis_aug_cfg["color_jitter"]["p"],
                same_on_batch=True,
                ),
            K.RandomGaussianBlur(
                kernel_size=tuple(vis_aug_cfg["gaussian_blur"]["kernel_size"]),
                sigma=tuple(vis_aug_cfg["gaussian_blur"]["sigma"]),
                p=vis_aug_cfg["gaussian_blur"]["p"],
                same_on_batch=True,
                ),
            K.RandomAffine(
                degrees=vis_aug_cfg["random_affine"]["degrees"],
                translate=tuple(vis_aug_cfg["random_affine"]["translate"]),
                p=vis_aug_cfg["random_affine"]["p"],
                same_on_batch=True,
                )
        )

        # define full pipeline for both training and inference
        self.img_tf_inference = torch.nn.Sequential(base_tf_pre)
        self.img_tf_train = torch.nn.Sequential(base_tf_pre, train_tf)

        # extract dataset features to be removed
        self.skip_features = []
        for v in self.feature_groups["REMOVE"]:
            self.skip_features.append(v)

    def joint_vel_transforms(self, v):
        noise = torch.randn_like(v[:, :1, :]) * self.joint_vel_std_dev   # v: (B, N_h, D) → noise: (B, 1, D) 
        return v + noise
    
    def joint_torque_transforms(self, v):
        noise = torch.randn_like(v[:, :1, :]) * self.joint_torque_std_dev   
        return v + noise

    @staticmethod
    def gripper_state_continuous2discrete(value, gripper_half_width=0.037):

        # input could be tensor or normal float
        if torch.is_tensor(value):
            return (value < gripper_half_width).float() # returns 0 (open) or 1 (close)
        else:
            return float(value < gripper_half_width)    
        
    @staticmethod
    def gripper_action_continuous2discrete(value, gripper_half_width=0.5):

        # input could be tensor, numpy array or normal float
        if torch.is_tensor(value):
            return (value > gripper_half_width).float() # returns 0 (open) or 1 (close)
        elif isinstance(value, np.ndarray):
            return (value > gripper_half_width).astype(np.float32)
        else:
            return float(value > gripper_half_width)    

    def transform(self, sample):

        state_parts = []
        action_parts = []
        
        for k, v in sample.items(): # each sample contains the N_h dimension (but no B dimension)
            
            if k in self.feature_groups["VISUAL"]:
                
                v = v.to(torch.float32) # convert data to tensor float32

                pre_shape = v.shape[:-3]  # it could be (B, N_h) or (B,)

                # flatten temporal dimension
                v_flat = v.reshape(-1, *v.shape[-3:])  # (*, C, H, W)

                # apply augmentations
                v_aug = self.img_tf_train(v_flat) if self.train else self.img_tf_inference(v_flat)
                v_aug = torch.zeros_like(v_aug) # TEMP FOR KINEMATICS ONLY TEST
                
                # reshape back
                v_aug = v_aug.reshape(*pre_shape, *v_aug.shape[-3:])

                # ensure images always have time dimension
                if v_aug.dim() == 4:  # (B, C, H, W)
                    v_aug = v_aug.unsqueeze(1)  # (B, 1, C, H, W)

                sample[k] = v_aug

            if k in self.feature_groups["STATE"]:

                v = v.to(torch.float32) # convert data to tensor float32

                for state_name in self.include_states:

                    if state_name == "q": 
                        part = v[..., self.state_slices["q"]]

                    elif state_name == "qdot": # add noise on joint velocities
                        part = v[..., self.state_slices["qdot"]]
                        part = self.joint_vel_transforms(part) if self.train else part

                    elif state_name == "tau": # add noise on joint torques
                        part = v[..., self.state_slices["tau"]]
                        part = self.joint_torque_transforms(part) if self.train else part

                    elif state_name == "fext": 
                        part = v[..., self.state_slices["fext"]]               

                    elif state_name == "ee_pose_absolute": 
                        pos = v[..., self.state_slices["ee_pos"]]
                        quat = v[..., self.state_slices["ee_quaternion"]]
                        quat = quat[..., [3,0,1,2]] # Dataset (x,y,z,w) → PyTorch3D (w,x,y,z)
                        # quat = standardize_quaternion(quat)
                        if self.orientation_type == "axis_angle":
                            ori = quaternion_to_axis_angle(quat)
                        elif self.orientation_type == "6D":
                            ori = matrix_to_rotation_6d(quaternion_to_matrix(quat))
                        elif self.orientation_type == "quaternion":
                            ori = quat
                        part = torch.cat([pos, ori], dim=-1)

                    elif state_name == "ee_pose_relative":
                        pos = v[..., self.state_slices["ee_pos"]]
                        quat = v[..., self.state_slices["ee_quaternion"]]
                        quat = quat[..., [3,0,1,2]] # Dataset (x,y,z,w) → PyTorch3D (w,x,y,z)
                        # quat = standardize_quaternion(quat)
                        R = quaternion_to_matrix(quat)

                        # base = last observed pose --> used both for STATE and ACTION
                        p_base = pos[:, -1:, :]      # (B,1,3)
                        R_base = R[:, -1:, :, :]     # (B,1,3,3)

                        # transform absolute ee_poses in relative ee_poses wrt last observed state
                        part = get_relative_poses_wrt_last_state(pos, R, p_base, R_base) # 6D notation for orientation

                    elif state_name == "gripper": # convert to discrete gripper state (0.0 or 1.0)
                        gripper_cont = v[..., self.state_slices["gripper"]]
                        part = self.gripper_state_continuous2discrete(gripper_cont) 

                    # add part to state vector
                    state_parts.append(part)

                v_new = torch.cat(state_parts, dim=-1) # (B, N_h, D)
                sample[k] = v_new


        for k, v in sample.items(): # each sample contains the N_h dimension (but no B dimension)

            if k in self.feature_groups["ACTION"]:

                v = v.to(torch.float32) # convert data to tensor float32

                for action_name in self.include_actions:

                    if action_name == "ee_pose_absolute":
                        pos = v[..., self.action_slices["ee_pos"]]
                        quat = v[..., self.action_slices["ee_quaternion"]]
                        quat = quat[..., [3,0,1,2]]  # Dataset (x,y,z,w) → PyTorch3D (w,x,y,z)
                        # quat = standardize_quaternion(quat)
                        if self.orientation_type == "axis_angle":
                            ori = quaternion_to_axis_angle(quat)
                        elif self.orientation_type == "6D":
                            ori = matrix_to_rotation_6d(quaternion_to_matrix(quat))
                        elif self.orientation_type == "quaternion":
                            ori = quat
                        part = torch.cat([pos, ori], dim=-1)

                    if action_name == "ee_pose_relative":

                        pos = v[..., self.action_slices["ee_pos"]]
                        quat = v[..., self.action_slices["ee_quaternion"]]
                        quat = quat[..., [3,0,1,2]] # Dataset (x,y,z,w) → PyTorch3D (w,x,y,z)
                        # quat = standardize_quaternion(quat)
                        R = quaternion_to_matrix(quat)

                        # transform absolute ee_poses in relative ee_poses wrt last observed state
                        part = get_relative_poses_wrt_last_state(pos, R, p_base, R_base) # 6D notation for orientation

                    elif action_name == "gripper": # convert to discrete gripper state (0.0 or 1.0)
                        gripper_cont = v[..., self.action_slices["gripper"]]
                        part = self.gripper_action_continuous2discrete(gripper_cont) 

                    # add part to state vector
                    action_parts.append(part)

                v_new = torch.cat(action_parts, dim=-1) # (B, N_h+N_c, D)

                if self.use_past_actions:
                    past_actions = v_new[:, :self.N_history, :]  # (B, N_h, D)

                future_actions = v_new[:, self.N_history:, :] # (B, N_c, D)
                sample[k] = future_actions


        # Manually remove extra length in "action_is_pad" created by LeRobot
        sample["action_is_pad"] = sample["action_is_pad"][:,self.N_history:]

        # append past actions to state if requested
        if self.use_past_actions:
            state_ft_name = self.feature_groups["STATE"][0]
            sample[state_ft_name] = torch.cat([sample[state_ft_name], past_actions], dim=-1)

        # drop unwanted features
        sample = {k: v for k, v in sample.items() if k not in self.skip_features}

        # print("transform", {k:v.shape for k,v in sample.items() if isinstance(v, torch.Tensor)})

        return sample