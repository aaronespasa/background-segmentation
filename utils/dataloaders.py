from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from dataset.dataset import ADE20KDataset

def setup_data_loaders(batch_size, num_workers=3, pin_memory=True, custom_ade20k_root_dir=None, image_height=256, image_width=256):
    ade20k_root_dir = ADE20K_ROOT_DIR if custom_ade20k_root_dir is None else custom_ade20k_root_dir
    train_dataset = ADE20KDataset(ade20k_root_dir, SCENE_CATEGORIES_PATH, "training", True, image_height, image_width)
    val_dataset = ADE20KDataset(ade20k_root_dir, SCENE_CATEGORIES_PATH, "validation", True, image_height, image_width)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader