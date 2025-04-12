import numpy as np
from .utils import relu, relu_grad

class ThreeLayerNN:
    def __init__(self, input_size, hidden_size, output_size, activation='relu', reg=0.0):  # 隐藏层参数减为1个
        self.params = {
            'W1': np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size),
            'b1': np.zeros(hidden_size),
            'W2': np.random.randn(hidden_size, output_size) * np.sqrt(2.0/hidden_size),
            'b2': np.zeros(output_size)
        }
        self.activation = activation
        self.reg = reg

    def forward(self, X):
        # 隐藏层
        z1 = X.dot(self.params['W1']) + self.params['b1']
        a1 = relu(z1)
        
        # 输出层
        scores = a1.dot(self.params['W2']) + self.params['b2']
        return scores, (X, z1, a1)

    def backward(self, scores, y, cache):
        X, z1, a1 = cache
        num_samples = X.shape[0]
        
        # 计算交叉熵梯度
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        dscores = probs
        dscores[range(num_samples), y] -= 1
        dscores /= num_samples

        # 输出层梯度
        grads = {
            'W2': a1.T.dot(dscores) + self.reg * self.params['W2'],
            'b2': np.sum(dscores, axis=0)
        }

        # 隐藏层梯度
        dhidden = dscores.dot(self.params['W2'].T)
        dhidden[z1 <= 0] = 0  # ReLU梯度

        grads['W1'] = X.T.dot(dhidden) + self.reg * self.params['W1']
        grads['b1'] = np.sum(dhidden, axis=0)
        
        return grads

    def predict(self, X):
        scores, _ = self.forward(X)
        return np.argmax(scores, axis=1)