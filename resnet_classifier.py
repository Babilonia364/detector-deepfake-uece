import torchvision.models as models
import torch.nn as nn

def get_resnet18():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to('cuda') #or cpur or preffered device
    return model
