# 辅助函数（单隐藏层版本）
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
    
    # 处理极端情况（避免log(0)）
    probs = np.clip(probs, 1e-10, 1.0)  # 限制概率范围

    # 计算交叉熵损失
    correct_logprobs = -np.log(probs[range(num_samples), y])
    loss = np.sum(correct_logprobs) / num_samples
    
    # 添加L2正则化项（单隐藏层版本）
    loss += 0.5 * reg * (np.sum(params['W1']**2) + 
                        np.sum(params['W2']**2))  # 移除了W3
    
    # 计算梯度 dscores (反向传播入口)
    dscores = probs.copy()
    dscores[range(num_samples), y] -= 1
    dscores /= num_samples
    
    return loss, dscores

def compute_accuracy(y_pred, y_true):
    """计算分类准确率"""
    return np.mean(y_pred == y_true)

def gradient_check(model, X_batch, y_batch, epsilon=1e-8):
    """
    数值梯度验证工具（单隐藏层版本）
    """
    # 强制使用 float64 精度
    X_batch = X_batch.astype(np.float64)
    y_batch = y_batch.astype(np.int64)
    for key in model.params:
        model.params[key] = model.params[key].astype(np.float64)

    # 获取解析梯度
    scores, cache = model.forward(X_batch)
    analytic_grads = model.backward(scores, y_batch, cache)
    
    # 验证参数列表（单隐藏层版本）
    param_keys = ['W1', 'b1', 'W2', 'b2']  # 移除了W3和b3

    for key in param_keys:
        if key not in model.params:
            continue
        
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
            
            # 计算误差
            abs_error = abs(numeric_grad - analytic_grad)
            rel_error = abs_error / (abs(numeric_grad) + abs(analytic_grad) + 1e-15)

            # 容错处理
            if model.activation == 'sigmoid':
                if max(abs(numeric_grad), abs(analytic_grad)) < 1e-4:
                    assert abs_error < 1e-6, f"Sigmoid小梯度检查失败，绝对误差：{abs_error:.6f}"
                else:
                    assert rel_error < 1e-3, f"相对误差过大：{rel_error:.6f}"
            else:
                assert rel_error < 1e-4, f"常规梯度检查失败：{rel_error:.6f}"

def enable_debug_mode(model):
    model.debug = True
