import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def ensure_results_directory():
    """
    确保结果文件夹存在
    """
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def plot_confusion_matrix(all_labels, all_preds, save_dir=None):
    """
    绘制混淆矩阵
    """
    if save_dir is None:
        save_dir = ensure_results_directory()
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    print("\n混淆矩阵已保存为 results/confusion_matrix.png")

def plot_cross_validation_accuracy(fold_results, save_dir=None):
    """
    绘制各折精度对比图
    """
    if save_dir is None:
        save_dir = ensure_results_directory()
    
    avg_acc = sum(fold_results) / len(fold_results)
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(fold_results)+1), fold_results, color='skyblue')
    plt.axhline(y=avg_acc, color='red', linestyle='--', label=f'Average: {avg_acc:.4f}')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Cross-Validation Accuracy per Fold')
    plt.ylim(0.9, 1.0)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'cross_validation_accuracy.png'))
    print("各折精度对比图已保存为 results/cross_validation_accuracy.png")

def plot_model_comparison(ensemble_acc, lr_scores, rf_scores, svm_scores, save_dir=None):
    """
    绘制模型性能对比图
    """
    if save_dir is None:
        save_dir = ensure_results_directory()
    
    model_names = ['集成模型', '逻辑回归', '随机森林', 'SVM']
    model_accuracies = [ensemble_acc, lr_scores.mean(), rf_scores.mean(), svm_scores.mean()]
    model_std = [0, lr_scores.std(), rf_scores.std(), svm_scores.std()]
    
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, model_accuracies, yerr=model_std, capsize=5, color=['green', 'blue', 'orange', 'purple'])
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.ylim(0.4, 1.0)
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
    print("模型性能对比图已保存为 results/model_comparison.png")

def plot_label_distribution(all_labels, save_dir=None):
    """
    绘制数据标签分布图
    """
    if save_dir is None:
        save_dir = ensure_results_directory()
    
    plt.figure(figsize=(8, 6))
    label_counts = pd.Series(all_labels).value_counts().sort_index()
    label_counts.plot(kind='bar', color='lightgreen')
    plt.xlabel('Depression Level')
    plt.ylabel('Count')
    plt.title('Label Distribution')
    plt.xticks(range(5), ['No Depression (0)', 'Mild (1)', 'Moderate (2)', 'Moderate-Severe (3)', 'Severe (4)'])
    plt.savefig(os.path.join(save_dir, 'label_distribution.png'))
    print("数据标签分布图已保存为 results/label_distribution.png")

def plot_roc_curve(all_labels, all_probs, save_dir=None):
    """
    绘制ROC曲线并计算AUC值
    """
    if save_dir is None:
        save_dir = ensure_results_directory()
    
    # 将标签二值化
    n_classes = len(np.unique(all_labels))
    y_test = label_binarize(all_labels, classes=range(n_classes))
    
    # 计算每类的ROC曲线和AUC值
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], np.array(all_probs)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算宏平均ROC曲线和AUC值
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average ROC curve (area = {roc_auc["macro"]:.4f})', linewidth=2)
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1, label=f'Class {i} (area = {roc_auc[i]:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    print("ROC曲线已保存为 results/roc_curve.png")
    
    # 保存AUC值到文件
    with open(os.path.join(save_dir, 'auc_values.txt'), 'w') as f:
        f.write("AUC Values:\n")
        for i in range(n_classes):
            f.write(f"Class {i}: {roc_auc[i]:.4f}\n")
        f.write(f"Macro-average: {roc_auc['macro']:.4f}\n")
    print("AUC值已保存为 results/auc_values.txt")

def plot_metrics_comparison(metrics, save_dir=None):
    """
    绘制模型评价指标对比图
    """
    if save_dir is None:
        save_dir = ensure_results_directory()
    
    plt.figure(figsize=(10, 6))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.plot(kind='bar', width=0.8)
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Model Evaluation Metrics')
    plt.ylim(0.8, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'))
    print("模型评价指标对比图已保存为 results/metrics_comparison.png")
