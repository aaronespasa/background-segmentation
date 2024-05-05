# pylint: disable=too-many-instance-attributes
"""
PyTorch ADE20K dataset.
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
        if self.split in ["training", "validation"]:
            self.images_dir = os.path.join(self.root_dir, "original", self.split)
            self.masks_dir = os.path.join(self.root_dir, "matting", self.split)
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
            A.Resize(self.image_width, self.image_height),
            # A.ShiftScaleRotate(shift_limit=0, scale_limit=-0.4, rotate_limit=0, p=0.5),
            # A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
            # A.GaussianBlur(p=0.2),
            ToTensorV2()
        ])

    def get_val_transformations(self):
        """Get validation transformations for data augmentation."""
        return A.Compose([
            A.Resize(self.image_width, self.image_height),
            ToTensorV2()
        ])

    def _read_image(self, image_name:str):
        """Returns the image as a numpy array in black and white format."""
        image_path = os.path.join(self.images_dir, image_name + ".jpg")
        image = np.array(Image.open(image_path).convert("L"))
        return image

    def _read_mask(self, image_name:str):
        """Returns the mask as a numpy array in RGB format with 32 classes."""
        mask_path = os.path.join(self.masks_dir, image_name + ".png")
        mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.float32)
        return mask

    def __len__(self):
        return len(self.files)

    def _create_color_mapping(self):
        # Mapeo de colores a índices de clase
        color_mapping = {
            "#F8502A": 0, "#506FBE": 1, "#8A2AA2": 2, "#3F2A7A": 3, "#04AE9A": 4,
            "#1CE3BE": 5, "#7E81C8": 6, "#87D5DE": 7, "#5903E2": 8, "#1269D3": 9,
            "#C81E3A": 10, "#04C386": 11, "#3ACB41": 12, "#70DBCF": 13, "#FF66F2": 14,
            "#B2DE3B": 15, "#7711C0": 16, "#339968": 17, "#1EAAEB": 18, "#215BA6": 19,
            "#F5616B": 20, "#2A7E64": 21, "#F056E3": 22, "#7E66E1": 23, "#926AD2": 24,
            "#E59C1F": 25, "#278B8A": 26, "#B54CE6": 27, "#52CBC9": 28, "#F0BD70": 29,
            "#307E6C": 30, "#3D1DBF": 31
        }

        # PIL.Image allows us to convert a color to an image in RGB format:
        # Image.new("RGB", (1, 1), color).convert("RGB") -> (1, 1, 3)

        # Therefore, we convert the color to an image and then we get the pixel value.
        # This way, we can get the RGB representation of the color in order to compare it
        # with the actual pixel values of the mask.
        return {tuple(np.array(Image.new("RGB", (1, 1), color).convert("RGB"))[0, 0]): idx for color, idx in color_mapping.items()}

    def _mask_to_class(self, mask):
        class_mask = torch.zeros((32, *mask.shape[:2]), dtype=torch.float32)
        for rgb_values, class_idx in self.color_to_class.items():
            class_mask[class_idx][mask == rgb_values] = 1.0
        return class_mask

    @staticmethod
    def rgb_to_hex(rgb_colors):
        hex_colors = []
        for rgb in rgb_colors:
            # Convert each RGB value to a hex string and format it
            hex_color = "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            hex_colors.append(hex_color)
        return hex_colors

    def print_unique_colors(self, mask):
        """Extract and return unique RGB colors from a mask."""
        # Reshape mask to a 2D array where each row is an RGB color
        reshaped_mask = mask.reshape(-1, 3)
        # Find unique colors along the rows
        unique_colors = np.unique(reshaped_mask, axis=0)

        hex_colors = self.rgb_to_hex(unique_colors)

        return hex_colors

    def __getitem__(self, index):
        image_name = self.files[index]
        image, mask = self._read_image(image_name), self._read_mask(image_name)

        hex_colors = self.print_unique_colors(mask)

        if (len(hex_colors) > 30):
            print(f"Image name: {image_name}")
            print(f"Length: {len(hex_colors)}")
            print(f"Hex colors: {hex_colors}")

        if self.transform:
            transformed = self.transformation(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
        
        mask = self._mask_to_class(mask)
        image = image.float() / 255.0

        return image, mask

if __name__ == "__main__":
    DATA_FOLDER = "data"
    
    train_dataset = BackgroundDataset(DATA_FOLDER, "training", True, image_height=512, image_width=256)

    # print the shape of a random image and its mask
    for i in range(400):
        image, mask = train_dataset[i]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")

    
