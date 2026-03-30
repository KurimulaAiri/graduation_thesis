import torch
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

def test(model, loader, criterion, device, verbose=False):
    """
    测试模型
    """
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            # 计算概率
            probs = torch.softmax(out, dim=1).cpu().numpy()
            all_probs.extend(probs)
    
    accuracy = correct / len(loader.dataset)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    if verbose:
        print("\n分类报告:")
        print(classification_report(all_labels, all_preds, zero_division=0))
        print("\n混淆矩阵:")
        print(confusion_matrix(all_labels, all_preds))
    
    return total_loss / len(loader), accuracy, precision, recall, f1, all_labels, all_preds, all_probs

def benchmark_models(X, y):
    """
    运行基准模型进行对比
    """
    print("\n========== 基准模型对比 ==========")
    
    # 逻辑回归
    lr = LogisticRegression(max_iter=1000)
    lr_scores = cross_val_score(lr, X, y, cv=5, scoring='accuracy')
    print(f"逻辑回归: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")
    
    # 随机森林
    rf = RandomForestClassifier(n_estimators=100)
    rf_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    print(f"随机森林: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
    
    # SVM
    svm = SVC(kernel='rbf')
    svm_scores = cross_val_score(svm, X, y, cv=5, scoring='accuracy')
    print(f"SVM: {svm_scores.mean():.4f} ± {svm_scores.std():.4f}")
    
    return lr_scores, rf_scores, svm_scores
