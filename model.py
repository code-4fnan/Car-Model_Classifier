"""Pretrained Transfer Learning Model for Car Classification."""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

def create_model(num_classes: int):
    """Loads a pretrained ResNet-18 and replaces the classifier head."""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    
    # Freeze the base layers to speed up training
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the fully connected layer to output our specific number of classes.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model, weights.transforms()

if __name__ == "__main__":
    model, transform = create_model(num_classes=9)
    sample = torch.randn(2, 3, 224, 224)
    output = model(sample)
    print("Input shape:", tuple(sample.shape))
    print("Output shape:", tuple(output.shape), "(batch size, num_classes)")