import os
from PIL import Image
import numpy as np
import torch

# Usage example: get_sample("1", validation=True, device="cuda")
def get_sample(filename, validation=True, device=None, go_parent_folder=False, width=256, height=256):
    split_folder = "validation" if validation else "training"
    data_folder = os.path.join("..", "data") if go_parent_folder else "data"
    image_path = os.path.join(data_folder, "images", split_folder, f"{filename}.jpg")
    mask_path = os.path.join(data_folder, "annotations", split_folder, f"{filename}.png")

    image = np.array(Image.open(image_path).convert("L").resize((width, height)))
    mask = np.array(Image.open(mask_path).convert("RGB").resize((width, height)))

    image_input, mask_input = None, None

    if device is not None:
        image_input = image.astype(np.float32)
        image_input = torch.tensor(image_input).unsqueeze(-1)
        image_input = image_input.permute(2, 0, 1).unsqueeze(0) # [N, C, H, W]
        image_input = image_input.to(device=device)

        mask_input = mask.astype(np.float32)
        mask_input = torch.tensor(mask_input)
        mask_input = mask_input.unsqueeze(0).unsqueeze(0) # [N, C, H, W]
        mask_input = mask_input.to(device=device)

    return image, mask, image_input, mask_input