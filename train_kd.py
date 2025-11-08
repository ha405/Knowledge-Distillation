import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Optional

class KDLoss(nn.Module):
    def __init__(self, ce_criterion, alpha=0.6, T=3.0):
        super(KDLoss, self).__init__()
        self.ce_criterion = ce_criterion
        self.alpha = alpha
        self.T = T

    def forward(self, teacher_logits, student_logits, labels):
        ce_loss = self.ce_criterion(student_logits, labels)
        log_ps = F.log_softmax(student_logits / self.T, dim=1)
        pt = F.softmax(teacher_logits / self.T, dim=1)
        kd_loss = F.kl_div(log_ps, pt, reduction='batchmean') * (self.T ** 2)
        total = (1.0 - self.alpha) * ce_loss + self.alpha * kd_loss
        return total, ce_loss.detach(), kd_loss.detach()

def train_kd(student, teacher, train_loader, optimizer, kd_criterion, epochs,
             device='cuda', scheduler=None, save_path: Optional[str]=None, log_every: Optional[int]=None):

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    teacher.to(device)
    student.to(device)

    teacher.eval()
    student.train()

    # For plotting curves
    epoch_losses = []
    epoch_ce_losses = []
    epoch_kd_losses = []
    epoch_top1 = []
    epoch_top5 = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        running_ce = 0.0
        running_kd = 0.0
        total_examples = 0
        correct_top1 = 0
        correct_top5 = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Student forward
            student_logits = student(inputs)

            # KD loss
            total_loss, ce_val, kd_val = kd_criterion(teacher_logits, student_logits, labels)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Statistics
            running_loss += total_loss.item() * batch_size
            running_ce += ce_val.item() * batch_size
            running_kd += kd_val.item() * batch_size
            total_examples += batch_size

            # Top-1 / Top-5
            _, pred_top5 = student_logits.topk(5, dim=1, largest=True, sorted=True)
            correct_top1 += (pred_top5[:, 0] == labels).sum().item()
            correct_top5 += pred_top5.eq(labels.view(-1, 1)).sum().item()

            if log_every and (batch_idx + 1) % log_every == 0:
                avg_loss_sofar = running_loss / total_examples
                avg_ce_sofar = running_ce / total_examples
                avg_kd_sofar = running_kd / total_examples
                top1_sofar = 100.0 * correct_top1 / total_examples
                top5_sofar = 100.0 * correct_top5 / total_examples
                print(f"Epoch {epoch} Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss {avg_loss_sofar:.4f} (CE {avg_ce_sofar:.4f}, KD {avg_kd_sofar:.4f}) | "
                      f"Top1 {top1_sofar:.2f}% Top5 {top5_sofar:.2f}%")

        # Scheduler step
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

        # Per-epoch metrics
        avg_loss = running_loss / total_examples
        avg_ce = running_ce / total_examples
        avg_kd = running_kd / total_examples
        top1 = 100.0 * correct_top1 / total_examples
        top5 = 100.0 * correct_top5 / total_examples

        epoch_losses.append(avg_loss)
        epoch_ce_losses.append(avg_ce)
        epoch_kd_losses.append(avg_kd)
        epoch_top1.append(top1)
        epoch_top5.append(top5)

        epoch_time = time.time() - epoch_start
        print("-" * 90)
        print(f"Epoch {epoch}/{epochs} finished in {epoch_time:.1f}s | "
              f"Avg Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | KD: {avg_kd:.4f} | "
              f"Train Top1: {top1:.2f}% Top5: {top5:.2f}%")
        print("-" * 90)

        # Optional checkpoint
        if save_path:
            ckpt = {
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(ckpt, f"{save_path}.epoch{epoch}.pth")

    # Final save
    if save_path:
        torch.save({'student_state_dict': student.state_dict()}, save_path)
        print(f"Saved final student checkpoint to: {save_path}")

    return student, epoch_losses, epoch_ce_losses, epoch_kd_losses, epoch_top1, epoch_top5
