import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def train(model, loader, optimizer, criterion, device):
    """
    训练模型
    """
    model.train()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
    
    accuracy = correct / len(loader.dataset)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return total_loss / len(loader), accuracy, precision, recall, f1
