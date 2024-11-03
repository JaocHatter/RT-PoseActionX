import torch
import torch.optim as optim
import numpy as  np
from tqdm import tqdm

def adjust_learning_rate(optimizer, epoch, batch_idx, num_batches, base_lr, warm_up_epoch, lr_ratio, num_epoch):
    if epoch < warm_up_epoch:
        lr = base_lr * (epoch + 1) / warm_up_epoch
    else:
        T_max = num_batches * (num_epoch - warm_up_epoch)
        T_cur = num_batches * (epoch - warm_up_epoch) + batch_idx

        eta_min = base_lr * lr_ratio
        lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + np.cos((T_cur / T_max) * np.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def load_optimizer(optimizer_type, model, base_lr, weight_decay, nesterov=False):
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=0.9,
            nesterov=nesterov,
            weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=base_lr,
            weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported optimizer type")
    return optimizer

def evaluate(model, test_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    num_batches = len(test_loader)
    with torch.no_grad():
        for data, label in tqdm(test_loader):
            data = data.float().to(device)
            label = label.long().to(device)

            output = model(data)
            loss = loss_fn(output, label)

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == label).sum().item() / label.size(0)
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    print(f'Evaluation, Loss: {avg_loss:.4f}, Accuracy: {avg_acc*100:.2f}%')
    return avg_acc

def train(model, optimizer, train_loader, epoch, num_epoch, base_lr, warm_up_epoch, lr_ratio, loss_fn, device):
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = len(train_loader)
    for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
        adjust_learning_rate(optimizer, epoch, batch_idx, num_batches, base_lr, warm_up_epoch, lr_ratio, num_epoch)

        data = data.float().to(device)
        label = label.long().to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == label).sum().item() / label.size(0)
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {avg_loss:.4f}, Accuracy: {avg_acc*100:.2f}%')
