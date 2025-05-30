import torchvision.models as models
import torch.nn as nn

def get_resnet18(num_classes=2):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to('cuda') #or cpur or preffered device
    return model
