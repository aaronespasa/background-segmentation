from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from dataset.dataset import BackgroundDataset

def setup_data_loaders(batch_size, num_workers=3, pin_memory=True, image_height=512, image_width=384):
    train_dataset = BackgroundDataset("data", "training", True, image_height, image_width)
    val_dataset = BackgroundDataset("data", "validation", True, image_height, image_width)

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