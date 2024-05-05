# pylint: disable=too-many-instance-attributes
"""
PyTorch Dataset class for the Matting Human Small Dataset.
"""

import torch
import os
from glob import glob
import numpy as np
from PIL import Image
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class BackgroundDataset(Dataset):
    """
    PyTorch Dataset class for the following Kaggle dataset:
    https://www.kaggle.com/datasets/aaronespasa/matting-human-small-dataset
    """

    def __init__(self, root_dir:str, split:str, transform:bool=True, image_height=512, image_width=256):
        """
        :param root_dir (string): Directory with all the images.
        :param transform (callable, optional): Optional transform to be applied on a sample.
        """
        [setattr(self, arg, value) for arg, value in locals().items() if arg != 'self'] # set the arguments as attributes
        self._set_files()

    def _set_files(self):
        """Fill the list of files depending on the dataset type (split)"""
        if self.split in ["training", "validation", "test"]:
            self.images_dir = os.path.join(self.root_dir, self.split, "original")
            self.masks_dir = os.path.join(self.root_dir, self.split, "matting")
            self.files = [os.path.basename(path).split(".")[0] for path in glob(self.images_dir + "/*.jpg")]
            self.transformation = self.get_train_transformations() if self.split == "training" else self.get_val_transformations()
        else:
            raise ValueError(
                f"The split value must be \'training\' or \'validation\', not \'{self.split}\'.")

    def get_train_transformations(self):
        """Get training transformations for data augmentation.
        mean: [0.5394, 0.4728, 0.4129] & std: [0.0035, 0.0037, 0.0042]
        """
        return A.Compose([
            A.Resize(self.image_height, self.image_width),
            # A.ShiftScaleRotate(shift_limit=0, scale_limit=-0.4, rotate_limit=0, p=0.5),
            # A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
            # A.GaussianBlur(p=0.2),
            ToTensorV2()
        ])

    def get_val_transformations(self):
        """Get validation transformations for data augmentation."""
        return A.Compose([
            A.Resize(self.image_height, self.image_width),
            ToTensorV2()
        ])

    def _read_image(self, image_name:str):
        """Returns the image as a numpy array in black and white format."""
        image_path = os.path.join(self.images_dir, image_name + ".jpg")
        image = np.array(Image.open(image_path).convert("RGB"))
        return image

    def _read_mask(self, image_name:str):
        """Returns the mask as a numpy array in black and white format."""
        mask_path = os.path.join(self.masks_dir, image_name + ".png")
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 0).astype(np.uint8) * 255
        return mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_name = self.files[index]
        image, mask = self._read_image(image_name), self._read_mask(image_name)

        if self.transform:
            transformed = self.transformation(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
        
        image = image.float() / 255.0
        mask = mask.float() / 255.0
        mask = mask.unsqueeze(0)

        return image, mask

if __name__ == "__main__":
    DATA_FOLDER = os.path.join("..", "data")
    
    # width: 600; height: 800
    # width: 384; height: 512
    # [3, H:512, W:384]; [1, H:512, W:384]
    train_dataset = BackgroundDataset(DATA_FOLDER, "training", True, image_height=512, image_width=384)

    image, mask = train_dataset[0]

    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")

    
