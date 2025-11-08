import torch
import torch.nn as nn

def train(model, loader, optimizer, criterion, epochs, device='cuda'):
    model.to(device)
    model.train()

    epoch_losses, epoch_top1, epoch_top5 = [], [], []

    for epoch in range(epochs):
        total_loss, correct_top1, correct_top5, total = 0.0, 0, 0, 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, pred_top5 = outputs.topk(5, 1)
            total += labels.size(0)
            correct_top1 += (pred_top5[:, 0] == labels).sum().item()
            correct_top5 += pred_top5.eq(labels.view(-1, 1)).sum().item()

        avg_loss = total_loss / total
        top1_acc = 100 * correct_top1 / total
        top5_acc = 100 * correct_top5 / total

        epoch_losses.append(avg_loss)
        epoch_top1.append(top1_acc)
        epoch_top5.append(top5_acc)

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Top-1: {top1_acc:.2f}% | Top-5: {top5_acc:.2f}%")

    return epoch_losses, epoch_top1, epoch_top5


def evaluate(model, loader, criterion, device='cuda'):
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct_top1, correct_top5, total = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            _, pred_top5 = outputs.topk(5, dim=1, largest=True, sorted=True)
            total += labels.size(0)
            correct_top1 += (pred_top5[:, 0] == labels).sum().item()
            correct_top5 += pred_top5.eq(labels.view(-1, 1)).sum().item()

    avg_loss = total_loss / total
    top1_acc = 100 * correct_top1 / total
    top5_acc = 100 * correct_top5 / total

    print(f"Eval Loss: {avg_loss:.4f} | Top-1 Acc: {top1_acc:.2f}% | Top-5 Acc: {top5_acc:.2f}%")
    return avg_loss, top1_acc, top5_acc
