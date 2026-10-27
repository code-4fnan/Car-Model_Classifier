"""Pretrained Transfer Learning Model."""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

def create_model(num_classes: int):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    
    # Base layers remain unfrozen so the model can learn car-specific features
        
    # Replace final classification head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model, weights.transforms()