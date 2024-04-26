import os

from time import time
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.cuda.amp import GradScaler, autocast
import numpy as np

import wandb

from .callbacks import save_model, early_stopping
from .epochinfo import log_epoch_info
from .diceloss import DiceLoss
from .iouloss import IoULoss
from .metrics import plot_metrics
from .onebatch import train_one_batch

def train_fn(train_loader, model, optimizer, criterion, scaler, scheduler, iou_loss, dice_loss, train_losses, train_iou, train_dice_scores, elapsed_times, epoch, device, is_edge_needed):
    train_running_loss = 0.0
    train_dice_score = 0.0
    train_iou_score = 0.0
    start_time = time()
    
    model.train()
    for image, mask in tqdm(train_loader, desc=f"Training (epoch {epoch})"):
        image, mask = image.to(device=device, non_blocking=True), mask.to(device=device, non_blocking=True)
        
        optimizer.zero_grad()

        # forward
        with autocast():
            output = model(image)
            loss = criterion(output, mask)
        
        # backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_running_loss += loss.item()
        train_iou_score += iou_loss(output, mask, sigmoid=True).item()
        train_dice_score += dice_loss(output, mask, sigmoid=True).item()
    
    # update scheduler with the mean loss of the epoch
    scheduler.step(train_running_loss / len(train_loader))
    
    train_losses.append(train_running_loss / len(train_loader))
    train_iou.append(train_iou_score / len(train_loader))
    train_dice_scores.append(train_dice_score / len(train_loader))
    elapsed_times.append(time() - start_time)

def val_fn(val_loader, model, criterion, iou_loss, dice_loss, val_losses, val_iou, val_dice_scores, epoch, device, is_edge_needed):
    val_running_loss = 0.0
    val_dice_score = 0.0
    val_iou_score = 0.0
    
    model.eval()
    with torch.no_grad():
        for image, mask, edge in tqdm(val_loader, desc=f"Validation (epoch {epoch})"):
            image, mask = image.to(device=device, non_blocking=True), mask.to(device=device, non_blocking=True)
            if is_edge_needed: edge = edge.to(device=device, non_blocking=True)

            # forward
            output = model(image)
            loss = criterion(output, mask)

            val_running_loss += loss.item()
            val_iou_score += iou_loss(output, mask, sigmoid=True).item()
            val_dice_score += dice_loss(output, mask, sigmoid=True).item()

    val_losses.append(val_running_loss / len(val_loader))
    val_iou.append(val_iou_score / len(val_loader))
    val_dice_scores.append(val_dice_score / len(val_loader))

def fit(epochs:int, model, train_loader, val_loader, criterion:nn, optimizer:torch.optim, scheduler, epoch_val, device, is_edge_needed, wandb, wandb_table, model_name="UNet") -> dict:
    torch.cuda.empty_cache()
    dice_loss, iou_loss, scaler = DiceLoss(), IoULoss(), GradScaler()
    train_losses, val_losses, train_iou, val_iou, train_dices, val_dices, elapsed_times = [], [], [], [], [], [], []
    min_loss, not_improved = np.inf, 0


    for epoch in range(epochs):
        epoch_val[0] += 1

        train_fn(train_loader, model, optimizer, criterion, scaler, scheduler, iou_loss, dice_loss, train_losses, train_iou, train_dices, elapsed_times, epoch+1, device, is_edge_needed)
        
        val_fn(val_loader, model, criterion, iou_loss, dice_loss, val_losses, val_iou, val_dices, epoch+1, device, is_edge_needed)

        log_epoch_info(epoch+1, epochs, epoch_val[0], model, wandb, wandb_table, train_losses, val_losses, train_iou, val_iou, train_dices, val_dices, elapsed_times, show_edge=is_edge_needed, device=device)
        
        if (early_stopping(val_losses[-1], min_loss, not_improved, patience=5)[0]):
            break

        if epoch_val[0] % 5 == 0:
            save_model(model, epoch_val[0], val_dices[-1], model_name=model_name)
    
    if epoch_val[0] % 5 != 0:
        save_model(model, epoch_val[0], val_dices[-1], model_name=model_name)
    
    history = { "epochs": list(range(1, epochs+1)), "train_losses": train_losses, "val_losses": val_losses, "train_dices": train_dices, "val_dices": val_dices }
    return history

def finish_training(model, wandb_table, history):
    # Log the table to WandB
    wandb.log({"predictions": wandb_table})

    wandb.finish()

    if history is not None:
        plot_metrics(history, metrics={'Loss': 'losses', 'Dice': 'dices'})

    # empty the cache to avoid CUDA out of memory error
    torch.cuda.empty_cache()

    # free the GPU memory of the model
    del model

def get_model_num_parameters(model):
    return round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2)

def train_model(
        model,
        learning_rate,
        num_epochs,
        train_loader,
        val_loader,
        batch_size,
        device,
        weight_decay=None,
        img_h=256,
        img_w=256,
        train_one_batch=False,
        is_edge_needed=False,
        model_name="fastvitwithfpn",
        run_name="Run Name"
    ):
    model = model.to(device)

    # Weights and biases initialization:
    os.environ.update({'WANDB_NOTEBOOK_NAME': 'training.ipynb'})
    wandb.init(project="teeth-segmentation",
               entity="uc3m-ml",
               config={"learning_rate": learning_rate, "batch_size": batch_size, "epochs": num_epochs,
                       "model_name": model_name, "image_dims": f"{img_h}x{img_w}", "num_parameters(millions)": get_model_num_parameters(model),},
               name=run_name
    )

    columns = ["ID", "Image", "Mask"] + (["Edge"] if is_edge_needed else []) + ["Prediction"]
    wandb_table = wandb.Table(columns=columns)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) if weight_decay else Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    wandb.watch(model, criterion, log="all")

    history = None
    try: 
        if train_one_batch:
            train_one_batch(model, train_loader, optimizer, criterion, epochs=200, device=device) # (set a small batch size first)
        else:
            epoch_val = [0] # to keep track of the epoch number in case the fit function is called multiple times
            history = fit(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, epoch_val, device, is_edge_needed, wandb, wandb_table, model_name=model_name)
    except KeyboardInterrupt:
        finish_training(model, wandb_table, history)

    finish_training(model, wandb_table, history)