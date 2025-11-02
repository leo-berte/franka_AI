import torchvision.transforms as T

# Add state augmentation noise:
# joint positions: σ ≈ 0.002–0.01 rad (≈ 0.1°–0.6°)
# joint speeds: σ ≈ 0.02 rad/s
# torques: σ ≈ 0.02 rad/s 
# pose (x,y,z): σ ≈ 0.5–2 mm
# orientation (axis-angle): σ ≈ 0.5–2° 
# gripper: nothing to do

def get_transforms(train=True):

    if train:
        return T.Compose([
            T.Resize((224, 224)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3),
            T.RandomApply([T.RandomAffine(degrees=5, translate=(0.05, 0.05))], p=0.5),
            T.ConvertImageDtype(torch.float32), # convert from uint8 [0,255] to tensor [0,1]
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]), # convert to [-1,1]
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])