import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - BCE_EXP) ** self.gamma * BCE

        return focal_loss

class FGBGLoss(nn.Module):
    def __init__(self, criterion, lambda_fg=0.3, lambda_bg=2):
        super(FGBGLoss, self).__init__()
        self.criterion = criterion
        self.lambda_fg = lambda_fg
        self.lambda_bg = lambda_bg

    def forward(self, inputs, targets):
        reverse_targets = targets.clone()
        reverse_targets[reverse_targets == 1] = -1
        reverse_targets[reverse_targets == 0] = 1
        reverse_targets[reverse_targets == -1] = 0
        bg_loss = self.criterion(inputs * reverse_targets, torch.zeros_like(inputs))
        fg_loss = self.criterion(inputs * targets, targets)
        fgbg_loss = self.lambda_fg * fg_loss + self.lambda_bg * bg_loss

        return fgbg_loss
