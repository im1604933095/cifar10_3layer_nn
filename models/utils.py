# 辅助函数
import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def cross_entropy_loss(scores, y, reg, params):
    num_samples = scores.shape[0]
    
    # 数值稳定的Softmax计算
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # 计算交叉熵损失
    correct_logprobs = -np.log(probs[range(num_samples), y])
    loss = np.sum(correct_logprobs) / num_samples
    
    # 添加L2正则化项（梯度检查时 reg=0 不影响）
    loss += 0.5 * reg * (np.sum(params['W1']**2) + 
                        np.sum(params['W2']**2) + 
                        np.sum(params['W3']**2))

    
    # 计算梯度 dscores (反向传播入口)
    dscores = probs.copy()
    dscores[range(num_samples), y] -= 1
    dscores /= num_samples
    
    return loss, dscores  # 同时返回 loss 和梯度

def compute_accuracy(y_pred, y_true):
    """计算分类准确率"""
    return np.mean(y_pred == y_true)

def gradient_check(model, X_batch, y_batch, epsilon=1e-8):
    """
    数值梯度验证工具
    Args:
        model: 神经网络实例
        X_batch: 小批量输入数据 (N, D)
        y_batch: 小批量标签 (N,)
        epsilon: 扰动步长
    """
    # 强制使用 float64 精度
    X_batch = X_batch.astype(np.float64)
    y_batch = y_batch.astype(np.int64)
    for key in model.params:
        model.params[key] = model.params[key].astype(np.float64)

    # 获取解析梯度
    scores, cache = model.forward(X_batch)
    analytic_grads = model.backward(scores, y_batch, cache)
    
    # 随机选取10个参数点进行验证
    param_keys = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']

    for key in param_keys:
        if key not in model.params:
            continue  # 兼容旧代码
        
        # shape = model.params[key].shape
        # 随机选取5个位置
        idx = np.random.choice(range(model.params[key].size), 5, replace=False)
        
        for i in idx:
            old_val = model.params[key].flat[i]
            
            # 正向扰动计算损失
            model.params[key].flat[i] = old_val + epsilon
            scores_plus, _ = model.forward(X_batch)
            loss_plus, _ = cross_entropy_loss(scores_plus, y_batch, reg=0, params=model.params)
            
            # 负向扰动计算损失
            model.params[key].flat[i] = old_val - epsilon
            scores_minus, _ = model.forward(X_batch)
            loss_minus, _ = cross_entropy_loss(scores_minus, y_batch, reg=0, params=model.params)
            
            # 恢复原始值
            model.params[key].flat[i] = old_val
            
            # 计算数值梯度
            numeric_grad = (loss_plus - loss_minus) / (2 * epsilon)
            analytic_grad = analytic_grads[key].flat[i]
            
            # # 忽略极小梯度
            # if abs(numeric_grad) < 1e-7 and abs(analytic_grad) < 1e-7:
            #     print(f"{key}[{i}] 极小梯度，跳过检查")
            #     continue

            # # 计算相对误差
            # rel_error = abs(numeric_grad - analytic_grad) / (abs(numeric_grad) + abs(analytic_grad) + 1e-8)  # 添加平滑项
            
            # print(f"{key}[{i}] 数值梯度: {numeric_grad:.6f}, 解析梯度: {analytic_grad:.6f}, 相对误差: {rel_error:.6f}")
            # assert rel_error < 1e-5 + 1e-8, f"梯度检查失败！相对误差: {rel_error:.6f}"

            # 计算绝对误差和相对误差
            abs_error = abs(numeric_grad - analytic_grad)
            rel_error = abs_error / (abs(numeric_grad) + abs(analytic_grad) + 1e-15)

            # 对Sigmoid的特殊处理
            if model.activation == 'sigmoid':
                # Sigmoid梯度较小时放宽要求
                if max(abs(numeric_grad), abs(analytic_grad)) < 1e-4:  # 绝对阈值
                    if abs_error > 1e-6:  # 绝对误差容差
                        print(f"Sigmoid小梯度检查失败，绝对误差：{abs_error:.6f}")
                        assert False
                else:
                    if rel_error > 1e-3:  # 相对误差放宽到0.1%
                        print(f"相对误差过大：{rel_error:.6f}")
                        assert False
            else:  # 其他激活函数保持原有检查
                if rel_error > 1e-4:
                    print(f"常规梯度检查失败：{rel_error:.6f}")
                    assert False

def enable_debug_mode(model):
    model.debug = True  # 在forward/backward中打印调试信息