import matplotlib.pyplot as plt
from .callbacks import show_model_output
import torch
import torch.nn.functional as F

def train_one_batch(model, loader, optimizer, criterion, epochs=100, show_edge=False, device="cuda"):
    """Train the model for one batch."""
    data = next(iter(loader))
    images, masks, edges = data
    images = images.to(device)
    masks = masks.to(device)
    edges = edges.to(device)

    for epoch in range(1, epochs + 1):
        preds = model(images)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}/{epochs}, Loss: {loss.item():.4f}")
        if epoch % 20 == 0:
            show_model_output(model, show_edge=show_edge, device=device, opt_data={"images":images, "masks":masks, "edges":edges})

def train_one_batch_magicwall(edge_aware_model, enhanced_net_model, loader, optimizer, criterion, epochs=100, show_edge=False, device="cuda"):
    """Train the model for one batch."""
    data = next(iter(loader))
    images, masks, edges = data
    images = images.to(device)
    masks = masks.to(device)
    edges = edges.to(device)

    for epoch in range(1, epochs + 1):
        segmentation_map, edge_map = edge_aware_model(images)
        
        edges_resized = F.interpolate(edges, size=(128, 128), mode="nearest")

        # image.shape: [32, 3, 512, 512]
        edge_loss = criterion(edge_map, edges_resized) 
        segmentation_map_loss = criterion(segmentation_map, masks) # segmentation_map.shape: [32, 1, 512, 512]

        # enhanced_model_input = torch.cat((images, segmentation_map), dim=1)

        # output = enhanced_net_model(enhanced_model_input)

        # output_loss = criterion(output, masks)

        # loss = edge_loss + segmentation_map_loss + output_loss
        loss = edge_loss + segmentation_map_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}/{epochs}, Loss: {loss.item():.4f}")
        if epoch % 20 == 0:
            show_model_output(edge_aware_model, show_edge=show_edge, device=device, opt_data={"images":images, "masks":masks, "edges":edges}, enhanced_net_model=enhanced_net_model, model_is_magicwall=True)