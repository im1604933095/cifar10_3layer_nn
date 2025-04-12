# 测试脚本（适配单隐藏层）
import numpy as np
from models.neural_net import ThreeLayerNN
from train import load_data  # 确保从train.py导入load_data
import matplotlib.pyplot as plt
from models.utils import cross_entropy_loss

def load_test_data():
    """从CIFAR-10加载测试集数据"""
    _, _, _, _, X_test, y_test = load_data(data_dir='data/cifar-10-batches-py')
    return X_test, y_test

def test(weights_path, hidden_size=256):
    # 1. 加载测试数据
    X_test, y_test = load_test_data()
    
    # 2. 初始化模型（单隐藏层结构）
    model = ThreeLayerNN(
        input_size=3072, 
        hidden_size=hidden_size,  # 仅一个隐藏层参数
        output_size=10, 
        activation='relu'
    )
    
    # 3. 加载训练好的权重
    model.params = np.load(weights_path, allow_pickle=True).item()
    
    # 4. 预测并计算准确率
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)

    # 5. 计算测试损失
    test_scores, _ = model.forward(X_test)
    test_loss, _ = cross_entropy_loss(test_scores, y_test, reg=0, params=model.params)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {acc:.4f}')

if __name__ == '__main__':
    # 示例调用（需确保weights.npy是单隐藏层模型的权重）
    weights_path = 'outputs/weights.npy'
    
    # 测试模型性能
    test(weights_path=weights_path, hidden_size=512)  # 根据实际训练参数调整