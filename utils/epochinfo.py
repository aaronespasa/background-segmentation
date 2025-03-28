from .callbacks import show_model_output

def get_trajectory_emoji(array):
    """Get the emoji for the trajectory."""
    if len(array) < 2:
        return ""
    
    last_value, second_last_value = array[-1], array[-2]

    if last_value > second_last_value:
        return " (📈)"
    elif last_value < second_last_value:
        return " (📉)"
    else:
        return " (🔷)"

def log_epoch_info(epoch, num_epochs, epoch_val, model, wandb, wandb_table, train_losses, val_losses, train_iou, val_iou, elapsed_times, device):
    """Log the epoch information."""
    print(f"\nEpoch: {epoch}/{num_epochs}",
          f"➡️ Loss --> Training{get_trajectory_emoji(train_losses)}: {train_losses[-1]:.4f} | Validation{get_trajectory_emoji(val_losses)}: {val_losses[-1]:.4f}",
          f"➡️ IoU --> Training{get_trajectory_emoji(train_iou)}: {(train_iou[-1]*100):.2f}% | Validation{get_trajectory_emoji(val_iou)}: {(val_iou[-1]*100):.2f}%",
          sep="\n\t", end="\n\n")
    
    if epoch % 2 == 0:
        show_model_output(model, wandb, wandb_table, filename="0", device=device, epoch_val=epoch_val)
    
    wandb.log({
        "epoch": epoch,
        "train_loss": train_losses[-1],
        "val_loss": val_losses[-1],
        "train_iou": train_iou[-1],
        "val_iou": val_iou[-1],
        "train_elapsed_time": round(elapsed_times[-1], 2),
    })