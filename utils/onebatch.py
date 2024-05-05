import matplotlib.pyplot as plt
from .callbacks import show_model_output
import torch
import torch.nn.functional as F

def train_one_batch(model, loader, optimizer, criterion, epochs=100, device="cuda"):
    """Train the model for one batch."""
    data = next(iter(loader))
    images, masks = data
    images = images.to(device)
    masks = masks.to(device)

    for epoch in range(1, epochs + 1):
        preds = model(images)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}/{epochs}, Loss: {loss.item():.4f}")
        if epoch % 20 == 0:
            show_model_output(model, device=device, opt_data={"images":images, "masks":masks})
