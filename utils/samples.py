import os
from PIL import Image
import numpy as np
import torch

VALID_SPLIT_SET = ["training", "validation", "test"]

def get_sample(filename, split_set="validation", device=None, go_parent_folder=False, width=384, height=512):
    if split_set not in VALID_SPLIT_SET:
        raise ValueError(f"Invalid split_set value. Must be one of {VALID_SPLIT_SET}.")

    data_folder = os.path.join("..", "data") if go_parent_folder else "data"
    image_path = os.path.join(data_folder, split_set, "original", f"{filename}.jpg")
    mask_path = os.path.join(data_folder, split_set, "matting", f"{filename}.png")

    image = np.array(Image.open(image_path).convert("RGB").resize((width, height)))
    mask = np.array(Image.open(mask_path).convert("L").resize((width, height)))
    mask = mask > 0
    matting = np.array(Image.open(mask_path).convert("RGB").resize((width, height)))

    image_input, mask_input = None, None

    if device is not None:
        image_input = torch.tensor(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device=device) # [1, 3, H, W]
        mask_input = torch.tensor(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device=device) # [1, 1, H, W]
        image_input = image_input / 255.0
    
    mask = mask.astype(np.uint8) * 255

    return image, mask, matting, image_input, mask_input