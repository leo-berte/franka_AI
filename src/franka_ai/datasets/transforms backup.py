import torchvision.transforms as T
import torch

  

class CustomTransforms():

    """
    Class for applying custom transformations to data:
    - Image jitter, affine transform, resize
    - Add noise to proprioception
    """

    def __init__(self, train=False):
        
        self.train = train
        self.joint_pos_std_dev = 0.002 # σ ≈ 0.002–0.01 rad (≈ 0.1°–0.6°)
        self.joint_vel_std_dev = 0.002 # σ ≈ 0.02 rad/s
        self.joint_torques_std_dev = 0.002 # σ ≈ 0.02 rad/s 
        self.cart_pos_position_std_dev = 0.002 # σ ≈ 0.5–2 mm
        self.cart_pos_orientation_std_dev = 0.002 # (axis-angle): σ ≈ 0.5–2° 

        base_tf = [
                T.ConvertImageDtype(torch.float32), # convert from uint8 [0,255] to tensor [0,1]
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]), # convert to [-1,1]
            ]
    
        train_tf = [
                T.Resize((224, 224)),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3),
                T.RandomApply([T.RandomAffine(degrees=5, translate=(0.05, 0.05))], p=0.5)
            ]
        
        self.img_tf_train = T.Compose(base_tf + train_tf)
        self.img_tf_inference = T.Compose(base_tf)

    def joint_pos_transforms(self, v):
        noise = torch.randn_like(v) * self.joint_pos_std_dev
        return v + noise
    
    def joint_vel_transforms(self, v):
        noise = torch.randn_like(v) * self.joint_vel_std_dev
        return v + noise

    def transform(self, sample):
        
        for k, v in sample.items():
            
            # Images
            if "image" in k:
                    sample[k] = self.img_tf_train(v) if self.train else self.img_tf_inference(v)

            # Joint positions
            elif "observation.state" in k:
                    sample[k] = self.joint_pos_transforms(v) if self.train else v

            # Joint velocities
            elif "observation.joint_velocities" in k:
                    sample[k] = self.joint_vel_transforms(v) if self.train else v

        return sample