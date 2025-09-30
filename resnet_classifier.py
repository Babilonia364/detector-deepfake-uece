import torchvision.models as models
import torch.nn as nn
import torch

def get_model(model_type='resnet18', num_classes=2):
    """
    Função para criar modelos ResNet melhorados para detecção de deepfakes
    com regularização e arquitetura mais robusta
    
    Args:
        model_type (str): Tipo de modelo ('resnet18' ou 'resnet50')
    
    Returns:
        torch.nn.Module: Modelo ResNet
    """
    # Carregar modelo pré-treinado baseado no tipo
    if model_type.lower() == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_type.lower() == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Modelo {model_type} não suportado. Use 'resnet18' ou 'resnet50'")

    # # Congelar apenas as primeiras camadas
    # for name, param in model.named_parameters():
    #     if 'layer1' in name or 'layer2' in name:
    #         param.requires_grad = False

    # Modificar a camada final para 2 classes (real vs fake)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )

    # Mover para o dispositivo disponível (mais flexível)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to('cuda') #or cpur or preffered device
    return model
