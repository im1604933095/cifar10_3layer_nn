import itertools
import os
import csv
import numpy as np
from train import train

def hyperparameter_search():
    """超参数网格搜索函数"""
    # 定义完整的参数网格
    param_grid = {
        'lr': [0.1, 0.07, 0.05, 0.03],  # 增加0.07作为中间值
        'hidden_size': [896, 1024, 1152, 1280],  # 缩小范围，步长~128
        'reg': [0.02, 0.015, 0.012, 0.01],  # 细化正则化步长
        'batch_size': [128, 192, 256]  # 增加中间值192
    }
    # 固定搜索阶段的epochs
    SEARCH_EPOCHS = 5
    
    # 初始化最佳记录
    best_acc = 0.0
    best_params = {}
    
    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)
    
    # 初始化日志文件
    with open('outputs/search_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = list(param_grid.keys()) + ['best_val_acc']
        writer.writerow(header)
    
    # 生成参数组合（修正：使用param_grid.values()）
    param_combinations = itertools.product(*param_grid.values())
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    
    # 遍历所有组合
    for i, params in enumerate(param_combinations, 1):
        current_params = dict(zip(param_grid.keys(), params))
        print(f"\n=== Testing Combination {i}/{total_combinations}  Epochs: {SEARCH_EPOCHS} ===")
        print("Parameters:", current_params)
        current_params['epochs'] = SEARCH_EPOCHS  # 注入固定值
        
        # 训练并获取最佳验证准确率
        current_best_acc = train(**current_params, is_search=True)
        # 记录日志
        with open('outputs/search_log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(params) + [current_best_acc])
        
        # 更新全局最佳
        if current_best_acc > best_acc:
            best_acc = current_best_acc
            best_params = current_params.copy()
            del best_params['epochs']  # 删除epochs信息
            print(f"🔥 New Best! Acc: {best_acc:.4f}")
    
    # 保存最佳参数
    print("\n=== Best Configuration ===")
    print(f"Validation Accuracy: {best_acc:.4f}")
    print("Parameters:", best_params)
    
    with open('outputs/best_params.txt', 'w') as f:
        f.write(str(best_params))

if __name__ == '__main__':
    hyperparameter_search()