# 主训练脚本
import numpy as np
import os
from models.neural_net import ThreeLayerNN
from models.utils import cross_entropy_loss, compute_accuracy
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

# 余弦退火（比固定衰减更优）
def cosine_lr(epoch, max_epochs, base_lr, warmup_ratio=0.1):
    if epoch < max_epochs * warmup_ratio:
        return base_lr * (epoch + 1) / (max_epochs * warmup_ratio)  # 线性预热
    return 0.5 * base_lr * (1 + np.cos(np.pi * (epoch - max_epochs * warmup_ratio) / (max_epochs * (1 - warmup_ratio))))

def train(hidden_size=256, activation='relu',
         lr=0.01, reg=0.001, epochs=100, batch_size=256, patience=10, is_search=False):    # 加载数据
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
    initial_lr = lr  # 初始学习率（保持固定）
    train_losses = []
    val_losses = []
    val_acc_history = []
    
    # 训练循环
    for epoch in range(epochs):  # 未来可考虑把这一块拉到外面，作为一个函数train_one_epoch
        # Mini-batch训练
        indices = np.random.permutation(X_train.shape[0])
        epoch_loss = 0
        num_batches = 0
        # 计算当前学习率（始终基于initial_lr）
        if is_search:
            # 固定学习率衰减（每轮衰减5%）
            current_lr = initial_lr * (0.95 ** epoch)
        else:
            # 正式训练用余弦退火
            current_lr = cosine_lr(epoch, epochs, initial_lr)
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
                model.params[param] -= current_lr * grads[param]
            # model.params = {k: v - current_lr * grads[k] for k, v in model.params.items()}  # 向量化操作更快

            epoch_loss += loss
            num_batches += 1
        
        # 验证评估
        val_pred = model.predict(X_val)
        val_acc = compute_accuracy(val_pred, y_val)
        val_acc_history.append(val_acc)

        # 计算平均训练损失
        train_loss = epoch_loss / num_batches
        train_losses.append(train_loss)
        
        # 验证集评估（新增损失计算）
        val_scores, _ = model.forward(X_val)
        val_loss, _ = cross_entropy_loss(val_scores, y_val, model.reg, model.params)
        val_losses.append(val_loss)
        
        # if epoch % 5 == 0 or (epoch+1) % 5 == 0:  # 每5个epoch备份一次
        #     backup_path = f'outputs/backup_epoch{epoch+1}.npy'
        #     np.save(backup_path, model.params)
        #     print(f"Backup model at epoch {epoch+1} to {backup_path}")

        # 早停机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('outputs', exist_ok=True)  # 确保目录存在
            np.save('outputs/weights.npy', model.params)
            print(f"Best model updated at epoch {epoch+1}, Val Acc={best_val_acc:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience and epoch > 15 and epoch < epochs - 10:
                print(f"Early stopping at epoch {epoch}")
                break

        # 打印进度
        print(f"Epoch {epoch+1}/{epochs}: Cur LR={current_lr:.4f}, Train Loss={train_loss:.4f}, Val Loss={loss:.4f}, Val Acc={val_acc:.4f}, Best={best_val_acc:.4f}")
           
    if not is_search:
        # 可视化部分
        plt.figure(figsize=(15, 5))

        # 损失曲线 (左图)
        plt.subplot(131)
        plt.plot(train_losses, 'b-', linewidth=2, label='Train Loss')
        plt.plot(val_losses, 'r--', linewidth=2, label='Val Loss')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.title('Training & Validation Loss', fontsize=14)

        # 准确率曲线 (中图)
        plt.subplot(132)
        plt.plot(val_acc_history, 'g-', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim([0.3, 0.7])
        plt.title('Validation Accuracy', fontsize=14)

        # 权重分布 (右图)
        plt.subplot(133)
        weights = np.concatenate([model.params['W1'].flatten(), 
                                model.params['W2'].flatten()])
        plt.hist(weights, bins=50, color='purple', alpha=0.7)
        plt.axvline(x=0, color='k', linestyle='--', linewidth=1)
        plt.xlabel('Weight Value', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.title(f'Weight Distribution\n(μ={weights.mean():.3f}, σ={weights.std():.3f})', fontsize=14)
        plt.tight_layout()
        plt.savefig('outputs/training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 12))
        W1 = model.params['W1']
        for i in range(16):
            plt.subplot(4, 4, i+1)
            # 归一化到[0,1]并转换为RGB
            neuron = W1[:, i].reshape(32, 32, 3)
            neuron = (neuron - neuron.min()) / (neuron.max() - neuron.min() + 1e-8)
            plt.imshow(neuron)
            plt.axis('off')
            plt.title(f'Neuron {i+1}', fontsize=8)
        plt.suptitle('First Layer Weight Patterns', y=0.92, fontsize=16)
        plt.savefig('outputs/layer1_patterns.png', dpi=300)
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
    act = 'sigmoid'
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

def visualize_weights(weights_path, img_shape=(32, 32, 3)):
    """可视化第一层兼容字典或数组输入）"""
    # 加载权重
    weights = np.load(weights_path, allow_pickle=True)
    
    # 处理不同存储格式
    if isinstance(weights, np.ndarray):
        w1 = weights  # 直接是数组
    elif isinstance(weights, dict):
        w1 = weights['W1']  # 字典格式
    else:
        raise ValueError("Unsupported weights format")
    
    # 可视化
    plt.figure(figsize=(12, 6))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        channel = w1[:, i].reshape(img_shape)
        channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        plt.imshow(channel)
        plt.axis('off')
    plt.savefig('outputs/w1_visual.png')
    plt.close()

if __name__ == '__main__':
    # test_load_data()  # 先测试数据加载
    # grad_check()
    
    with open('outputs/best_params.txt') as f:
        best_params = eval(f.read())
    # 添加必要参数
    best_params.update({
        'epochs': 50,        # 正式训练使用更多epoch
        'lr': best_params['lr'] * 1.2,  # 适当提升初始学习率（因余弦退火会快速下降）
        'batch_size': int(best_params['batch_size'] * 1.5),   # 增大批量大小（同步调整学习率）
        'patience': 10,       # 增大早停耐心值
        'is_search': False   # 关闭搜索模式
    })
    # 启动正式训练
    train(**best_params)
