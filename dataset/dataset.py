# pylint: disable=too-many-instance-attributes
"""
PyTorch ADE20K dataset.
"""

import os
from glob import glob
import numpy as np
from PIL import Image
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class ADE20KDataset(Dataset):
    """
    PyTorch ADE20K dataset.
    """

    def __init__(self, root_dir:str, scene_categories:str, split:str, transform:bool=False, image_height=256, image_width=256):
        """
        :param root_dir (string): Directory with all the images.
        :param transform (callable, optional): Optional transform to be applied on a sample.
        """
        [setattr(self, arg, value) for arg, value in locals().items() if arg != 'self'] # set the arguments as attributes
        self._set_files()

    def _set_files(self):
        """Fill the list of files depending on the dataset type (split)"""
        if self.split in ["training", "validation"]:
            self.images_dir = os.path.join(self.root_dir, "images", self.split)
            self.masks_dir = os.path.join(self.root_dir, "backgroundAnnotations", self.split)
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
            # A.Resize(256, 256),
            # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            # A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),

            A.Resize(self.image_height, self.image_width),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
            A.GaussianBlur(p=0.2),
            A.CLAHE(p=0.1),

            # A.Normalize(
            #     mean=[0.5394, 0.4728, 0.4128],
            #     std=[0.2427, 0.2459, 0.2548],
            #     max_pixel_value=255.0,
            #     p=1.0
            # ),
            ToTensorV2()
        ])

    def get_val_transformations(self):
        """Get validation transformations for data augmentation."""
        return A.Compose([
            A.Resize(self.image_height, self.image_width),
            # A.Normalize(
            #     mean=[0.5404, 0.4707, 0.4120],
            #     std=[0.2474, 0.2500, 0.2570],
            #     max_pixel_value=255.0,
            #     p=1.0
            # ),
            ToTensorV2()
        ])

    def _read_image(self, image_name:str):
        """Returns the image as a numpy array in RGB."""
        image_path = os.path.join(self.images_dir, image_name + ".jpg")
        image = np.array(Image.open(image_path).convert("RGB"))
        return image

    def _read_mask(self, image_name:str):
        """Returns the mask as a numpy array in grayscale (with values 0 and 1)."""
        mask_path = os.path.join(self.masks_dir, image_name + ".png")
        mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.float32)
        return mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_name = self.files[index]
        image, mask, edge = self._read_image(image_name), self._read_mask(image_name), self._read_edge(image_name)
        
        if self.transform:
            transformed = self.transformation(image=image, mask=mask, edge=edge)
            image, mask, edge = transformed["image"], transformed["mask"], transformed["edge"]
        
        image = image.float() / 255.0
        mask = mask.float().unsqueeze(0)
        edge = edge.float().unsqueeze(0)

        return image, mask, edge

if __name__ == "__main__":
    DATA_FOLDER, ADE20K_TRAIN_VAL_FOLDER = "data", "ADEChallengeData2016"
    ADE20K_ROOT_DIR = os.path.join(DATA_FOLDER, ADE20K_TRAIN_VAL_FOLDER)
    SCENE_CATEGORIES_PATH = os.path.join(DATA_FOLDER, ADE20K_TRAIN_VAL_FOLDER, "sceneCategories.txt")
    
    train_dataset = ADE20KDataset(ADE20K_ROOT_DIR, SCENE_CATEGORIES_PATH, "training", True)

    # print the shape of a random image and its mask
    image, mask, edge = train_dataset[1]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Edge shape: {edge.shape}")