import numpy as np


class Parameter:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLU:
    def __init__(self):
        self.non_negative = None

    def forward(self, x):
        self.non_negative = x >= 0
        return x * self.non_negative

    def backward(self, d):
        return d * self.non_negative

    def parameters(self):
        return []


class Linear:
    def __init__(self, in_features, out_features):
        w_shape = (in_features, out_features)
        self.w = Parameter(np.random.normal(loc=0.0, scale=np.sqrt(2 / sum(w_shape)), size=w_shape))
        self.b = Parameter(np.zeros(out_features))
        self.x = None

    def forward(self, x):
        self.x = x.copy()
        return np.dot(self.x, self.w.value) + self.b.value

    def backward(self, grad):
        self.w.grad += np.dot(self.x.T, grad)
        self.b.grad += np.sum(grad, axis=0)
        return np.dot(grad, self.w.value.T)

    def parameters(self):
        return [self.w, self.b]


class Flattener:
    def __init__(self):
        self.x_shape = None

    def forward(self, x):
        self.x_shape, batch_size = x.shape, x.shape[0]
        return x.reshape(batch_size, -1)

    def backward(self, grad):
        return grad.reshape(self.x_shape)

    def parameters(self):
        return []


class Dropout:
    def __init__(self, p=0.5, train=True, grad_check=False):
        self.p = p
        self.train = train
        self.grad_check = grad_check
        self.mask = None

    def forward(self, x):
        if self.train:
            if self.mask is None or not self.grad_check:
                self.mask = np.random.binomial(1, self.p, size=x.shape) / self.p
            return x * self.mask
        return x

    def backward(self, grad):
        if self.train:
            return grad * self.mask
        return grad

    def parameters(self):
        return []


class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        w_shape = (kernel_size * kernel_size * in_channels, out_channels)
        self.w = Parameter(np.random.normal(loc=0.0, scale=np.sqrt(2 / sum(w_shape)), size=w_shape))
        self.b = Parameter(np.zeros(out_channels))

        self.x_padded = None
        self.x_height, self.x_width = None, None

    def forward(self, x):
        batch_size, self.x_height, self.x_width, channels = x.shape

        self.x_padded = np.pad(array=x, mode='constant',
                               pad_width=((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))

        out_height = self.x_height + 2 * self.padding - self.kernel_size + 1
        out_width = self.x_width + 2 * self.padding - self.kernel_size + 1
        result = np.zeros((batch_size, out_height, out_width, self.out_channels))

        for j in range(out_height):
            for i in range(out_width):
                x_reshaped = self.x_padded[:, j:j + self.kernel_size, i:i + self.kernel_size, :].reshape(batch_size, -1)
                result[:, j, i, :] = np.dot(x_reshaped, self.w.value) + self.b.value
        return result

    def backward(self, grad):
        batch_size, out_height, out_width, out_channels = grad.shape

        x_grad = np.zeros_like(self.x_padded)

        for j in range(out_height):
            for i in range(out_width):
                grad_ij = grad[:, j, i, :]
                x_reshaped = self.x_padded[:, j:j + self.kernel_size, i:i + self.kernel_size, :].reshape(batch_size, -1)
                self.w.grad += np.dot(x_reshaped.T, grad_ij)
                self.b.grad += grad_ij.sum(axis=0)
                x_ij_grad = np.dot(grad_ij, self.w.value.T)
                x_ij_grad = x_ij_grad.reshape(batch_size, self.kernel_size, self.kernel_size, self.in_channels)
                x_grad[:, j:j + self.kernel_size, i:i + self.kernel_size, :] += x_ij_grad
        return x_grad[:, self.padding:self.padding + self.x_height, self.padding:self.padding + self.x_width, :]

    def parameters(self):
        return [self.w, self.b]


class MaxPool2d:
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.x = None

    def forward(self, x):
        batch_size, height, width, channels = x.shape
        self.x = x.copy()

        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        result = np.zeros((batch_size, out_height, out_width, channels))

        for j in range(out_height):
            for i in range(out_width):
                x_slice = x[:, j * self.stride: j * self.stride + self.kernel_size,
                            i * self.stride: i * self.stride + self.kernel_size, :]
                result[:, j, i, :] = np.amax(x_slice, (1, 2))
        return result

    def backward(self, grad):
        batch_size, out_height, out_width, out_channels = grad.shape

        x_grad = np.zeros_like(self.x)

        for sample in range(batch_size):
            for channel in range(out_channels):
                for j in range(out_height):
                    for i in range(out_width):
                        grad_ij = grad[sample, j, i, channel]
                        x_slice = self.x[sample, j * self.stride:j * self.stride + self.kernel_size,
                                         i * self.stride:i * self.stride + self.kernel_size, channel]
                        mask = (x_slice == np.max(x_slice))
                        n_max = np.sum(mask)
                        if n_max > 1:
                            grad_ij /= 2
                        x_grad[sample, j * self.stride:j * self.stride + self.kernel_size,
                               i * self.stride:i * self.stride + self.kernel_size, channel] += grad_ij * mask
        return x_grad

    def parameters(self):
        return []
