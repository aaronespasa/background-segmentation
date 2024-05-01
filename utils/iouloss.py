import torch
from torch import nn

class IoULoss(nn.Module):
    """
    Computes the Intersection over Union (IoU) loss for binary image segmentation.

    Math:
        - IoUScore = intersection(X, Y) / union(X, Y)
    """

    def __init__(self):
        """Inherits from the nn.Module class."""
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: float=1e-6, softmax: bool=True) -> torch.Tensor:
        """
        Args:
            inputs: (torch.Tensor) Predicted segmentation mask.
            targets: (torch.Tensor) Ground truth segmentation mask.
            smooth: (float) Smoothing parameter for the IoU loss (to avoid division by zero).
            softmax: (bool) True if you want to apply the softmax function to the inputs before processing.
        
        Returns:
            loss: (torch.Tensor) IoU loss.
        """
        assert torch.is_tensor(inputs), "Inputs must be a torch.Tensor."
        assert inputs.size() == targets.size(), "Sizes from the inputs & targets must be equal."
        assert inputs.dim() == 4, "The inputs must be a 4D tensor (BATCHES, CHANNELS, H, W)."
        assert inputs.device == targets.device, "Inputs and targets must be in the same device."

        with torch.no_grad():
            if softmax:
                inputs = torch.softmax(inputs)
                # TODO: Use 32 classes

            # Flatten the input and target tensors (matrix -> array)
            inputs = inputs.view(-1)
            targets = targets.view(-1)

            # Compute intersection and union
            intersection = (inputs * targets).sum()
            total = (inputs + targets).sum()
            union = total - intersection

            # Compute the IoU score and obtain the IoU loss
            return (intersection + smooth) / (union + smooth)

def test():
    """Test the IoU Loss"""
    loss = IoULoss()
    inputs = torch.tensor([
        [1, 0, 0],
        [0, 0, 1],
        [1, 1, 0]
    ]).expand(1, 1, 3, 3).float() # Ensure inputs are float for sigmoid
    targets = torch.tensor([
        [0, 0, 0],
        [0, 0, 1],
        [1, 1, 1]
    ]).expand(1, 1, 3, 3).float() # Ensure targets are float
    # Intersection: 3, Union: 5 (total 8 - intersection 3)
    # IoU Score: 3 / 5 = 0.6
    output = loss(inputs, targets, sigmoid=False)
    print(f"Result: {output.item():.1f}; Expected: 0.6")

if __name__ == "__main__":
    test()