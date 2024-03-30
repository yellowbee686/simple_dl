import numpy as np

class Conv2D:
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        # 初始化卷积核和偏置项
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * 0.1
        self.bias = np.zeros(output_channels)
        self.input = None
        self.grad_weights = None
        self.grad_bias = None
    
    def forward(self, input):
        self.input = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)
        batch_size, channels, height, width = self.input.shape
        output_height = (height - self.kernel_size) // self.stride + 1
        output_width = (width - self.kernel_size) // self.stride + 1
        output = np.zeros((batch_size, self.weights.shape[0], output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                output[:, :, i, j] = np.sum(self.input[:, :, h_start:h_end, w_start:w_end, np.newaxis] * self.weights[np.newaxis, :, :, :], axis=(2, 3, 4)) + self.bias
        
        return output
    
    def backward(self, grad_output):
        # 初始化梯度张量
        grad_input = np.zeros_like(self.input)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

        _, _, output_height, output_width = grad_output.shape

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                # 计算grad_input
                grad_input[:, :, h_start:h_end, w_start:w_end] += np.sum(
                    grad_output[:, :, i, j][:, np.newaxis, np.newaxis, np.newaxis] * self.weights,
                    axis=0
                )

                # 计算grad_weights
                self.grad_weights += np.sum(
                    self.input[:, :, h_start:h_end, w_start:w_end, np.newaxis] * grad_output[:, :, i, j, np.newaxis, np.newaxis, np.newaxis, :],
                    axis=(0, 1)
                )

        # 计算grad_bias
        self.grad_bias += np.sum(grad_output, axis=(0, 2, 3))

        # 删除padding
        if self.padding > 0:
            grad_input = grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return grad_input

# 示例使用
input = np.random.randn(1, 1, 5, 5)  # batch_size=1, input_channels=1, height=5, width=5
conv = Conv2D(input_channels=1, output_channels=1, kernel_size=3, stride=1, padding=1)
output = conv.forward(input)
print("Forward output shape:", output.shape)

# 假设反向传播的梯度全为1
grad_output = np.ones_like(output)
grad_input = conv.backward(grad_output)
print("Backward output shape:", grad_input.shape)
