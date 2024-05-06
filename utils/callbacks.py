import torch
import os
import matplotlib.pyplot as plt
from typing import TypedDict, Union
from .samples import get_sample

class OptData(TypedDict):
    images: torch.Tensor
    masks: torch.Tensor

def convert_to_plotimg(img: torch.Tensor) -> torch.Tensor:
    return img.permute(1, 2, 0).cpu().detach().numpy()

def show_model_output(model, wandb=None, wandb_table=None, filename="0", device="cuda", opt_data:Union[OptData,None]=None, epoch_val=0):
    if opt_data is not None:
        image, mask = opt_data["images"][0], opt_data["masks"][0]
        plot_image, plot_mask = convert_to_plotimg(image), convert_to_plotimg(mask).squeeze(-1)
        image, mask = image.unsqueeze(0), mask.unsqueeze(0)
    else:
        plot_image, plot_mask, _, image, mask = get_sample(filename, device=device)
    
    model.eval()
    with torch.no_grad():
        output = model(image)

        output = torch.sigmoid(output)
        output = output[0].squeeze().detach().cpu().numpy()

        imgs = [plot_image, plot_mask, output]
        titles = ["Image", "Mask", "Prediction"]
        
        if wandb is not None:
            # Convert images to the format expected by WandB
            wandb_imgs = [wandb.Image(np_img, caption=title) for np_img, title in zip(imgs, titles)]
            row_data = [epoch_val] + wandb_imgs
            wandb_table.add_data(*row_data)
        
        # Plot the images
        fig, axes = plt.subplots(1, len(imgs), figsize=(20, 10))
        for ax, img, title in zip(axes, imgs, titles):
            ax.axis("off")
            ax.set_title(title)
            ax.imshow(img, cmap='gray' if title != "Image" else None)
        plt.show()

def save_model(model, epoch, iou_score, model_name="UNet"):
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    torch.save(model.state_dict(), f"checkpoints/{model_name}_epoch_{epoch}_iou_{(iou_score*100):.2f}%.pth")

def early_stopping(val_loss, min_loss, not_improved, patience=7):
    """Return True if the validation loss didn't improve for the last `patience` epochs."""
    min_loss = val_loss if val_loss < min_loss else min_loss
    not_improved = 0 if val_loss < min_loss else not_improved + 1
    stop = False

    if val_loss > min_loss:
        if not_improved == patience:
            print(f"Loss didn\'t decrease for {patience} times, Stopping Training (early stopping)...")
            stop = True
    
    return stop, min_loss, not_improved