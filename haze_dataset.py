from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
from configuration import both_transform, transform_only_input


class HazeDataset(Dataset):
    def __init__(self, haze_dir, clean_dir):
        self.haze_dir = haze_dir
        self.clean_dir = clean_dir
        self.haze_dir_files = sorted(os.listdir(haze_dir))
        self.clean_dir_files = sorted(os.listdir(clean_dir))

    def __len__(self):
        return len(self.haze_dir_files)

    def __getitem__(self, item):
        img_haze_file = self.haze_dir_files[item]
        img_clean_file = self.clean_dir_files[item]
        img_haze_path = os.path.join(self.haze_dir, img_haze_file)
        img_clean_path = os.path.join(self.clean_dir, img_clean_file)
        real = np.array(Image.open(img_haze_path))
        target = np.array(Image.open(img_clean_path))
        augmentations = both_transform(image=real, image0=target)

        real = augmentations["image"]
        target = augmentations["image0"]

        real = transform_only_input(image=real)["image"]
        target = transform_only_input(image=target)["image"]
        return real, target

