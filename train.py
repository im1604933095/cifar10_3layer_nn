# 主训练脚本
import numpy as np
import os
from models.neural_net import ThreeLayerNN
from models.utils import cross_entropy_loss, compute_accuracy
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse

os.environ["OMP_NUM_THREADS"] = "4"  # 根据CPU核心数调整（i5通常为4核）
os.environ["MKL_NUM_THREADS"] = "4"

def load_data(data_dir='data/cifar-10-batches-py', val_ratio=0.1):
    """加载CIFAR-10数据并划分验证集
    Args:
        data_dir: 数据目录路径
        val_ratio: 验证集占训练集的比例（默认0.1）
    Returns:
        X_train, y_train: 训练集数据和标签
        X_val, y_val: 验证集数据和标签
        X_test, y_test: 测试集数据和标签
    """
    # 1. 加载原始训练数据（合并所有批次）
    train_batches = [f'data_batch_{i}' for i in range(1,6)]
    X_train, y_train = [], []
    
    for batch_file in train_batches:
        with open(os.path.join(data_dir, batch_file), 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
            X_train.append(batch['data'])
            y_train.extend(batch['labels'])
    
    X_train = np.concatenate(X_train).astype('float32')
    y_train = np.array(y_train)
    
    # 2. 划分验证集（关键修正点：使用 val_ratio 参数）
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=val_ratio, 
        random_state=42
    )
    
    # 3加载测试数据
    with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
        test_batch = pickle.load(f, encoding='latin1')
        X_test = test_batch['data'].astype('float32')
        y_test = np.array(test_batch['labels'])
    
    # 4. 数据预处理（归一化）
    mean_pixel = X_train.mean(axis=0)
    std_pixel = X_train.std(axis=0) + 1e-8
    
    X_train = (X_train - mean_pixel) / std_pixel
    X_val = (X_val - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel
    
    return X_train, y_train, X_val, y_val, X_test, y_test

    # # 实际使用时替换为真实数据加载代码
    # X_train = np.random.randn(45000, 3072).astype(np.float32)
    # y_train = np.random.randint(0, 10, 45000)
    # X_val = np.random.randn(5000, 3072).astype(np.float32) 
    # y_val = np.random.randint(0, 10, 5000)
    # return X_train, y_train, X_val, y_val, None, None

def train(hidden_size=256, activation='relu',
         lr=0.01, reg=0.001, epochs=100, batch_size=256, patience=5):    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # 初始化模型（修正隐藏层参数）
    model = ThreeLayerNN(input_size=3072, 
                        hidden_size=hidden_size, 
                        output_size=10, 
                        activation=activation, 
                        reg=reg)
    
    # 训练状态跟踪
    best_val_acc = 0.0
    no_improve = 0
    train_losses = []
    val_acc_history = []
    
    # 训练循环
    for epoch in range(epochs):
        # Mini-batch训练
        indices = np.random.permutation(X_train.shape[0])
        for i in range(0, X_train.shape[0], batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
            
            # 前向传播
            scores, cache = model.forward(X_batch)
            loss, _ = cross_entropy_loss(scores, y_batch, model.reg, model.params)
            
            # 反向传播
            grads = model.backward(scores, y_batch, cache)
            
            # 参数更新
            for param in model.params:
                model.params[param] -= lr * grads[param]
        
        # 验证评估
        val_pred = model.predict(X_val)
        val_acc = compute_accuracy(val_pred, y_val)
        val_acc_history.append(val_acc)
        train_losses.append(loss)
        
        # 早停机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            np.save('outputs/weights.npy', model.params)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # 学习率衰减
        lr *= 0.95
        
        # 打印进度
        print(f"Epoch {epoch+1}/{epochs}: Loss={loss:.4f}, Val Acc={val_acc:.4f}, Best={best_val_acc:.4f}")
    
    # 保存训练曲线
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(val_acc_history, label='Val Acc')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('outputs/train_curve.png')
    plt.close()
    
    return best_val_acc  # 返回整个训练过程中的最佳验证准确率

# 测试数据加载
def test_load_data():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(val_ratio=0.1)
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    print(f"Unique labels: {np.unique(y_train)}")
    
def grad_check():
    """梯度验证流程"""
    # 加载小批量数据（100样本）
    X_train, y_train, _, _, _, _ = load_data()
    X_batch, y_batch = X_train[:2000], y_train[:2000]
    
    # 初始化临时模型（关闭正则化）
    act = 'relu'
    # 初始化模型参数（单隐藏层）
    model = ThreeLayerNN(
        input_size=3072, 
        hidden_size=256,  # 只指定1个隐藏层大小
        output_size=10,
        activation=act,
        reg=0.0
    )
    print(f"激活函数: {act}")
    
    # 运行梯度检查
    from models.utils import gradient_check
    gradient_check(model, X_batch, y_batch)
    print("✅ 梯度检查通过！")

if __name__ == '__main__':
    test_load_data()  # 先测试数据加载
    grad_check()
    # # 命令行参数解析
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--hidden_size1', type=int, default=256)
    # parser.add_argument('--hidden_size2', type=int, default=128)
    # parser.add_argument('--activation', type=str, default='relu')
    # parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--reg', type=float, default=0.001)
    # parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--batch_size', type=int, default=256)
    # parser.add_argument('--patience', type=int, default=5)
    # args = parser.parse_args()
    
    # # 启动训练
    # train(
    #     hidden_size1=args.hidden_size1,
    #     hidden_size2=args.hidden_size2,
    #     activation=args.activation,
    #     lr=args.lr,
    #     reg=args.reg,
    #     epochs=args.epochs,
    #     batch_size=args.batch_size,
    #     patience=args.patience
    # )