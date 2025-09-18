import torchvision.models as models
import torch.nn as nn
import torch

def get_resnet18():
    # Usar a sintaxe moderna com weights em vez de pretrained
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Modificar a camada final para 2 classes (real vs fake)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    # Mover para o dispositivo disponível (mais flexível)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to('cuda') #or cpur or preffered device
    return model
