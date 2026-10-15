"""Custom CNN for Supercar vs SUV image classification."""
import torch
from torch import nn


class CarCNN(nn.Module):
    """
    Input: RGB tensor shaped (N, 3, 224, 224).
    Exactly three Conv2d -> ReLU -> MaxPool2d blocks,
    then a 128-neuron fully connected layer and 2 output logits.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        if num_classes != 2:
            raise ValueError("CarCNN is designed for exactly two classes.")

        self.features = nn.Sequential(
            # Block 1: 3 x 224 x 224 -> 16 x 112 x 112
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 16 x 112 x 112 -> 32 x 56 x 56
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 32 x 56 x 56 -> 64 x 28 x 28
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 64 * 28 * 28 = 50,176 features for a 224 x 224 input.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = CarCNN()
    sample = torch.randn(2, 3, 224, 224)
    output = model(sample)
    print("Input shape:", tuple(sample.shape))
    print("Output shape:", tuple(output.shape), "(batch size, 2 classes)")
