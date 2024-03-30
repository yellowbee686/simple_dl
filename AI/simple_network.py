import numpy as np
import torch.nn as nn
# np风格的简单网络实现
# 作为网络中的一层时继承Module更合适
class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, grad):
        # grad对input求导的形式能够用output更简洁的表示，因此在sigmoid中缓存output来实现
        return self.output * (1 - self.output) * grad
    
class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.input = None
    
    def forward(self, x):
        self.input = x
        return np.maximum(x, 0)
    
    def backward(self, grad):
        grad_backward = grad.copy() # 调用np.copy()是shallow copy，但这里假设是简单array，每个元素是基本类型，调用shallow copy即可
        grad_backward[self.input <= 0] = 0
        return grad_backward
    
class Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Linear, self).__init__()
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros(output_dim)
        self.input = None
        # 求完backward后在update_weights()时按照该方式更新
        # weights = weights - learning_rate * self.grad_weights
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, x):
        self.input = x
        # [B, input_dim] * [input_dim, output_dim] + [output_dim]
        # 直接self.input * self.weight报错，shape无法auto spread
        return np.dot(self.input, self.weights) + self.bias
    
    def backward(self, grad):
        self.grad_weights = np.dot(self.input.T, grad)
        # [B, output_dim] -> [output_dim] 求loss时已平均过，这里直接求和
        self.grad_bias = np.sum(grad, axis=0)
        return np.dot(self.weights.T, grad)

class SimpleNetwork:
    def __init__(self):
        self.layer1 = Linear(10, 5)
        self.relu = ReLU()
        self.layer2 = Linear(5, 1)
        self.sigmoid = Sigmoid()
        
    def forward(self, x):
        x = self.layer1.forward(x)
        x = self.relu.forward(x)
        x = self.layer2.forward(x)
        x = self.sigmoid.forward(x)
        return x
    
    def backward(self, grad_output):
        grad_output = self.sigmoid.backward(grad_output)
        grad_output = self.layer2.backward(grad_output)
        grad_output = self.relu.backward(grad_output)
        grad_output = self.layer1.backward(grad_output)
        return grad_output

    def update_weights(self, learning_rate):
        self.layer1.weights -= learning_rate * self.layer1.grad_weights
        self.layer1.bias -= learning_rate * self.layer1.grad_bias
        self.layer2.weights -= learning_rate * self.layer2.grad_weights
        self.layer2.bias -= learning_rate * self.layer2.grad_bias


def train():
    # 使用网络
    net = SimpleNetwork()
    x = np.random.randn(1, 10)  # 一个输入样本
    y_true = np.array([[1]])  # 真实标签

    # 前向传播
    output = net.forward(x)

    # 计算损失（这里使用简单的均方误差）
    loss = np.mean((output - y_true) ** 2)

    # 反向传播损失梯度
    grad_loss = 2 * (output - y_true) / output.size
    net.backward(grad_loss)

    # 更新权重
    net.update_weights(0.001)