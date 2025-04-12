# 测试脚本
import numpy as np
from models.neural_net import ThreeLayerNN
from train import load_data  # 确保从train.py导入load_data
import matplotlib.pyplot as plt
from models.utils import cross_entropy_loss

def load_test_data():
    """从CIFAR-10加载测试集数据"""
    _, _, _, _, X_test, y_test = load_data(data_dir='data/cifar-10-batches-py')
    return X_test, y_test

def test(weights_path, hidden_size1=256, hidden_size2=128):
    # 1. 加载测试数据
    X_test, y_test = load_test_data()  # 关键修复点
    
    # 2. 初始化模型（必须与训练时的结构一致）
    model = ThreeLayerNN(
        input_size=3072, 
        hidden_size1=hidden_size1, 
        hidden_size2=hidden_size2, 
        output_size=10, 
        activation='relu'
    )
    
    # 3. 加载训练好的权重
    model.params = np.load(weights_path, allow_pickle=True).item()
    
    # 4. 预测并计算准确率
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f'Test Accuracy: {acc:.4f}')

    test_scores, _ = model.forward(X_test)
    test_loss = cross_entropy_loss(test_scores, y_test, reg=0, params=model.params)
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {acc:.4f}')

def visualize_weights(weights_path):
    """可视化第一层权重（可选）"""
    weights = np.load(weights_path, allow_pickle=True).item()
    W1 = weights['W1']
    
    # 可视化代码
    plt.figure(figsize=(12,8))
    for i in range(100):
        plt.subplot(10,10,i+1)
        w_img = W1[:,i].reshape(32,32,3)
        w_img = (w_img - w_img.min()) / (w_img.max() - w_img.min())
        plt.imshow(w_img)
        plt.axis('off')
    plt.savefig('outputs/weight_visualization.png')

if __name__ == '__main__':
    test(weights_path='outputs/weights.npy')  # 示例调用