import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        alpha: pode ser um escalar ou tensor [w0, w1] para pesos das classes
        gamma: fator de foco
        reduction: 'mean' ou 'sum'
        """
        super(FocalLoss, self).__init__() 
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probabilidade da classe correta
        
        # Calcular os pesos alpha para cada elemento do batch
        if isinstance(self.alpha, torch.Tensor):
            # Garantir que alpha está no mesmo dispositivo
            alpha = self.alpha.to(inputs.device)
            # Selecionar o alpha correto para cada target
            alpha_t = alpha[targets]
        else:
            alpha_t = self.alpha
            
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss