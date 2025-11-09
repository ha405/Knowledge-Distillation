import torch
import torch.nn as nn
import torch.nn.functional as F

class HintDist(nn.Module):
    def __init__(self, ce_criterion, alpha=0.5, beta=0.5, T=4.0, gamma=0.2):
        super(HintDist, self).__init__()
        self.ce_criterion = ce_criterion
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.T = T

    def forward(self, teacher_logits, student_logits, labels, teacher_feat=None, student_feat=None):
        ce_loss = self.ce_criterion(student_logits, labels)
        log_ps = F.log_softmax(student_logits / self.T, dim=1)
        pt = F.softmax(teacher_logits / self.T, dim=1)
        kd_loss = F.kl_div(log_ps, pt, reduction='batchmean') * (self.T ** 2)
        if self.gamma > 0.0 and teacher_feat is not None and student_feat is not None:
            hint_loss = 0.5 * torch.norm(student_feat - teacher_feat, p='fro')**2
        else:
            hint_loss = torch.tensor(0.0, device=student_logits.device)

        total_loss = self.alpha * ce_loss + self.beta * kd_loss + self.gamma * hint_loss
        return total_loss, ce_loss.detach(), kd_loss.detach(), hint_loss.detach()


class HintLoss(nn.Module):
    def __init__(self):
        super(HintLoss, self).__init__()

    def forward(self, teacher_feat, student_feat):
        hint_loss = 0.5 * torch.norm(student_feat - teacher_feat, p='fro')**2
        return hint_loss

def freeze_layers_following(model: nn.Module, last_trainable_layer_name: str):
    freeze = False
    for name, param in model.named_parameters():
        if freeze:
            param.requires_grad = False
        if name.startswith(last_trainable_layer_name):
            freeze = True
