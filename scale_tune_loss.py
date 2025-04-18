import torch
import torch.nn as nn

class AdaptiveHuberLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # Calcul des erreurs absolues
        abs_errors = torch.abs(y_true - y_pred)

        # Déterminer delta comme la médiane des erreurs absolues
        delta = torch.median(abs_errors).item()

        # Appliquer la formule de Huber Loss élément par élément
        huber_loss = torch.where(
            abs_errors < delta,
            0.5 * abs_errors.pow(2),  # Partie quadratique (MSE)
            delta * (abs_errors - 0.5 * delta)  # Partie linéaire (MAE)
        )

        # Réduction (moyenne, somme ou aucune)
        if self.reduction == 'mean':
            return huber_loss.mean()
        elif self.reduction == 'sum':
            return huber_loss.sum()
        else:
            return huber_loss  # Retourne la loss sans réduction

# Test avec des données (16, 4, 256, 256)
y_pred = torch.randn(16, 4, 256, 256)
y_true = torch.randn(16, 4, 256, 256)

loss_fn = AdaptiveHuberLoss()
loss = loss_fn(y_pred, y_true)

print("Adaptive Huber Loss:", loss.item())

class QuantileHuberLoss(torch.nn.Module):
    def __init__(self, quantiles, delta=1.0):
        super().__init__()
        self.quantiles = torch.tensor(quantiles).view(-1, 1, 1, 1, 1)  # Adapté à tes données
        self.delta = delta

    def forward(self, y_pred, y_true):
        errors = y_true.unsqueeze(0) - y_pred.unsqueeze(1)  # Broadcasting sur les quantiles
        huber_loss = torch.where(
            errors.abs() <= self.delta,
            0.5 * errors.pow(2),
            self.delta * (errors.abs() - 0.5 * self.delta)
        )
        loss = torch.max(self.quantiles * huber_loss, (self.quantiles - 1) * huber_loss)
        return loss.mean()

# Utilisation :
quantiles = [0.1, 0.5, 0.9]
loss_fn = QuantileHuberLoss(quantiles, delta=1.0)

loss = loss_fn(y_pred, y_true)
print("Quantile Huber Loss:", loss.item())


def huber_loss1(input, target, delta=1.0, reduction='mean'):
    return F.huber_loss(input, target, delta=delta, reduction=reduction)

def quantile_loss(q,y_pred, target) :
    # calculate quantile loss
    losses = []
    for i, q in enumerate(q):
        errors = target - y_pred
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
    losses = 2 * torch.cat(losses, dim=-1).mean()

    return losses