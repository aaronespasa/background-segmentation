import torch
from dataset.dataset import BackgroundDataset
from models.models import get_model
from utils import IoULoss, show_model_output
import os

def test_model(encoder_name="resnet50", decoder_name="unet", checkpoint_name="unet_resnet50_0.001_epoch_25_iou_96.64%.pth", split_set="test", HEIGHT=512, WIDTH=384, DEVICE="cuda"):
    # Load the dataset
    test_loader = BackgroundDataset("data", split_set, True, HEIGHT, WIDTH)

    # IoU Loss
    iou_loss = IoULoss()

    # Load the model
    model = get_model(encoder=encoder_name, decoder=decoder_name)
    checkpoint = torch.load(os.path.join("checkpoints", checkpoint_name))
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    
    # Compute the mean IoU
    total_iou = 0
    for i, (images, masks) in enumerate(test_loader):
        images = images.unsqueeze(0)
        masks = masks.unsqueeze(0)
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        outputs = model(images)
        iou = iou_loss(outputs, masks)
        total_iou += iou.item()

    print(f"Mean IoU: {(total_iou / len(test_loader)*100):.2f}%")

    # Show some results
    for i, (images, masks) in enumerate(test_loader):
        if i == 5:
            break
        # opt_data = {
        #     "images": images.unsqueeze(0).to(DEVICE),
        #     "masks": masks.unsqueeze(0).to(DEVICE)
        # }
        show_model_output(model, wandb=None, wandb_table=None, device=DEVICE, filename=str(i))