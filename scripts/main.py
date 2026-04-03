import os
import sys
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入配置
from config.config import MODEL_CONFIG, TRAIN_CONFIG, DEVICE, RESULTS_CONFIG
from config.data_config import DATA_CONFIG

# 导入数据处理模块
from src.data import process_all_subjects

# 导入训练评估模块
from src.train import create_dataset, extract_flatten_features, train, test, benchmark_models
from src.train.visualizer import ensure_results_directory, plot_confusion_matrix, plot_cross_validation_accuracy, plot_model_comparison, plot_label_distribution, plot_roc_curve, plot_metrics_comparison

# 导入模型模块
from src.models import EnsembleClassifier

def main():
    print("开始处理数据...")
    # 处理数据
    data_dir = DATA_CONFIG['data_dir']
    subjects = process_all_subjects(data_dir)
    print(f"处理完成，共加载 {len(subjects)} 个样本")
    
    # 创建数据集
    dataset = create_dataset(subjects)
    print(f"创建数据集完成，共 {len(dataset)} 个有效样本")
    
    if len(dataset) == 0:
        print("没有有效的训练数据")
        return
    
    # 检查数据分布
    labels = [data.y.item() for data in dataset]
    label_distribution = pd.Series(labels).value_counts().sort_index()
    print("\n数据标签分布:")
    print(label_distribution)
    
    # 使用配置的设备
    device = torch.device(DEVICE)
    print(f"\n使用设备: {device}")
    
    # 动态设置输入特征维度
    MODEL_CONFIG['in_channels'] = dataset[0].x.shape[1]
    print(f"输入特征维度: {MODEL_CONFIG['in_channels']}")
    
    # 确保结果文件夹存在
    results_dir = ensure_results_directory()
    
    # 执行k折交叉验证
    kfold = KFold(n_splits=TRAIN_CONFIG['k_folds'], shuffle=True, random_state=TRAIN_CONFIG['random_state'])
    fold_results = []
    all_labels = []
    all_preds = []
    all_probs = []
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f"\n========== Fold {fold+1}/{TRAIN_CONFIG['k_folds']} ==========")
        
        # 创建训练集和测试集
        train_subset = torch.utils.data.Subset(dataset, train_ids)
        test_subset = torch.utils.data.Subset(dataset, test_ids)
        
        # 检查训练集和测试集的分布
        train_labels = [data.y.item() for data in train_subset]
        test_labels = [data.y.item() for data in test_subset]
        print(f"训练集大小: {len(train_subset)}, 测试集大小: {len(test_subset)}")
        print(f"训练集标签分布: {pd.Series(train_labels).value_counts().sort_index()}")
        print(f"测试集标签分布: {pd.Series(test_labels).value_counts().sort_index()}")
        
        # 创建数据加载器
        train_loader = DataLoader(train_subset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=TRAIN_CONFIG['batch_size'])
        
        # 初始化集成模型
        model = EnsembleClassifier(
            MODEL_CONFIG['in_channels'], 
            MODEL_CONFIG['hidden_channels'], 
            MODEL_CONFIG['num_classes']
        ).to(device)
        print("集成模型初始化完成")
        
        # 计算类别权重，解决样本不均衡问题
    label_counts = pd.Series(labels).value_counts().sort_index()
    total_samples = len(labels)
    class_weights = torch.tensor(
        [total_samples / (len(label_counts) * label_counts.get(i, 1)) for i in range(MODEL_CONFIG['num_classes'])],
        dtype=torch.float
    ).to(device)
    print(f"类别权重: {class_weights.cpu().numpy()}")
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=TRAIN_CONFIG['learning_rate'], 
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        # 添加学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        print("优化器和损失函数设置完成")
        
        # 训练模型
        best_test_acc = 0
        best_labels = []
        best_preds = []
        best_probs = []
        patience_counter = 0
        
        print("开始训练模型...")
        for epoch in range(TRAIN_CONFIG['num_epochs']):
            train_loss, train_acc, train_prec, train_rec, train_f1 = train(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc, test_prec, test_rec, test_f1, test_labels, test_preds, test_probs = test(model, test_loader, criterion, device)
            
            # 更新学习率
            scheduler.step(test_loss)
            
            # 早停机制
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_labels = test_labels
                best_preds = test_preds
                best_probs = test_probs
                patience_counter = 0
                # 保存最佳模型
                fold_model_path = os.path.join(results_dir, f'depression_classifier_best_fold{fold+1}.pth')
                torch.save(model.state_dict(), fold_model_path)
                print(f"保存最佳模型 (Fold {fold+1}) 到 {fold_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= TRAIN_CONFIG['patience']:
                    print(f"早停：连续 {TRAIN_CONFIG['patience']} 轮测试精度未提升")
                    break
        
        fold_results.append(best_test_acc)
        all_labels.extend(best_labels)
        all_preds.extend(best_preds)
        all_probs.extend(best_probs)
        print(f"Fold {fold+1} 最佳测试精度: {best_test_acc:.4f}")
    
    # 计算平均精度
    avg_acc = sum(fold_results) / len(fold_results)
    print(f"\n========== 交叉验证结果 ==========")
    print(f"各折精度: {[f'{acc:.4f}' for acc in fold_results]}")
    print(f"平均精度: {avg_acc:.4f}")
    
    # 计算整体评估指标
    print("\n========== 整体评估结果 ==========")
    from sklearn.metrics import classification_report
    report = classification_report(all_labels, all_preds, zero_division=0)
    print(report)
    
    # 保存分类报告到文件
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    print("\n分类报告已保存为 results/classification_report.txt")
    
    # 生成可视化图表
    plot_confusion_matrix(all_labels, all_preds, results_dir)
    plot_cross_validation_accuracy(fold_results, results_dir)
    
    # 绘制ROC曲线和计算AUC值
    plot_roc_curve(all_labels, all_probs, results_dir)
    
    # 计算模型评价指标
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # 保存详细评价指标到文件
    with open(os.path.join(results_dir, 'model_metrics.txt'), 'w') as f:
        f.write("Model Evaluation Metrics:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision (Macro): {precision:.4f}\n")
        f.write(f"Recall (Macro): {recall:.4f}\n")
        f.write(f"F1 Score (Macro): {f1:.4f}\n")
        f.write(f"Average Cross-Validation Accuracy: {avg_acc:.4f}\n")
    print("模型评价指标已保存为 results/model_metrics.txt")
    
    # 绘制模型评价指标对比图
    metrics = {
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    }
    plot_metrics_comparison(metrics, results_dir)
    
    # 运行基准模型对比
    X, y = extract_flatten_features(dataset)
    lr_scores, rf_scores, svm_scores = benchmark_models(X, y)
    plot_model_comparison(avg_acc, lr_scores, rf_scores, svm_scores, results_dir)
    plot_label_distribution(all_labels, results_dir)
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(results_dir, 'depression_classifier.pth'))
    print("\n模型已保存为 results/depression_classifier.pth")
    print("\n所有结果已保存到 results 文件夹")

if __name__ == '__main__':
    main()
