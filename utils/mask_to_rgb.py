import numpy as np
from PIL import Image

def class_mask_to_rgb(class_mask):
    color_mapping = {
        "#F8502A": 0, "#506FBE": 1, "#8A2AA2": 2, "#3F2A7A": 3, "#04AE9A": 4,
        "#1CE3BE": 5, "#7E81C8": 6, "#87D5DE": 7, "#5903E2": 8, "#1269D3": 9,
        "#C81E3A": 10, "#04C386": 11, "#3ACB41": 12, "#70DBCF": 13, "#FF66F2": 14,
        "#B2DE3B": 15, "#7711C0": 16, "#339968": 17, "#1EAAEB": 18, "#215BA6": 19,
        "#F5616B": 20, "#2A7E64": 21, "#F056E3": 22, "#7E66E1": 23, "#926AD2": 24,
        "#E59C1F": 25, "#278B8A": 26, "#B54CE6": 27, "#52CBC9": 28, "#F0BD70": 29,
        "#307E6C": 30, "#3D1DBF": 31
    }

    class_to_color = {idx: np.array(Image.new("RGB", (1, 1), color).convert("RGB"))[0, 0] for color, idx in color_mapping.items()}

    # Initialize an empty RGB image
    rgb_mask = np.zeros((class_mask.shape[1], class_mask.shape[2], 3), dtype=np.uint8)

    # Iterate over each class index and set the corresponding color
    for class_idx in range(32):
        class_slice = class_mask[class_idx, :, :] 
        color = class_to_color[class_idx] 
        rgb_mask[class_slice == 1] = color

    return rgb_mask
