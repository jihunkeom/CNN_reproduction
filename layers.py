import numpy as np
np.random.seed(0)


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        self.grads[0][...] = dW
        return None


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
        self.name = "Affine"

    def forward(self, x):
        W, b = self.params
        self.x = x
        out = np.dot(x, W) + b
        return out

    def backward(self, dout):
        W, b = self.params
        db = np.sum(dout, axis=0)
        dW = np.dot(self.x.T, dout)
        dx = np.dot(dout, W.T)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Relu:
    def __init__(self):
        self.params, self.grads = [], []
        self.mask = None
        self.name = "Relu"

    def forward(self, x):
        mask = (x <= 0)
        x[mask] = 0
        self.mask = mask
        return x

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.name = "Dropout"

    def forward(self, x, train=False):
        self.train = train
        if train:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            x *= self.mask
        else:
            x *= (1.0 - self.dropout_ratio)
        return x

    def backward(self, dout):
        dout *= self.mask
        return dout


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, x, t):
        if x.ndim == 2:
            x = x.T
            x -= np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            y = y.T
        else:
            x -= np.max(x)
            y = np.exp(x) / np.sum(np.exp(x))

        if y.size == t.size:
            t = np.argmax(t, axis=1)

        if y.ndim == 1:
            y = y.reshape(1, -1)
            t = t.reshape(1, -1)

        batch_size = y.shape[0]
        loss = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

        self.cache = [y, t, batch_size]
        return loss

    def backward(self, dout=1):
        y, t, batch_size = self.cache
        dout = y.copy()
        dout[np.arange(batch_size), t] -= 1
        dout /= batch_size
        return dout


class Convolution:
    def __init__(self, W, b, stride, pad=0):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.stride = stride
        self.pad = pad
        self.x = None
        self.cache = None

    def add_pad(self, x):
        tmp = np.zeros((self.pad, x.shape[1]), dtype='f')
        return np.vstack([tmp, x, tmp])

    def forward(self, x):
        self.cache = []
        W, b = self.params
        if self.pad > 0:
            x = self.add_pad(x)
        self.x = x
        out_h = int((x.shape[0] - W.shape[0]) / self.stride) + 1
        out_w = 1

        out = np.empty((out_h, out_w), dtype="f")
        r = 0
        self.cache = []
        for i in range(out_h):
            tmp = x[r: r + W.shape[0]]
            out[i] = np.sum(tmp * W)
            self.cache.append(tmp)
            r += self.stride

        return out + b

    def backward(self, dout):
        x = self.x
        W, b = self.params
        dW, db = self.grads

        dW_tmp = np.zeros_like(dW)
        dx = np.zeros_like(x, dtype='f')

        r = 0
        for i in range(dout.shape[0]):
            dx[r: r + W.shape[0]] = W * dout[i]
            dW_tmp += dout[i] * self.cache[i]
            r += self.stride
        self.grads[0][...] = dW_tmp
        self.grads[1][...] = np.sum(dout)
        return dx


class Pooling:
    def __init__(self):
        self.params, self.grads = [], []
        self.x = None
        self.argmax = None

    def forward(self, x):
        self.x = x
        self.argmax = np.argmax(x)
        out = self.x.max()
        return out

    def backward(self, dout):
        dx = (np.zeros_like(self.x)).astype('f')
        dx[self.argmax] = dout

        return dx

class BatchConvolution:
    def __init__(self, W, b, stride, pad=0):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.stride = stride
        self.pad = pad
        self.x = None
        self.cache = None
        self.batch_size = None

    def forward(self, x):
        self.cache = []
        W, b = self.params
        self.x = x
        self.batch_size = x.shape[0]
        out_h = int((x.shape[1] - W.shape[0]) / self.stride) + 1
        out_w = 1
        out = np.empty((self.batch_size, out_h, out_w), dtype='f')

        r = 0
        self.cache = []
        for i in range(out_h):
            tmp = x[:, r:r+W.shape[0], :]
            for j in range(self.batch_size):
                out[j, i] = np.sum(tmp[j] * W)
            self.cache.append(tmp)
            r += self.stride
        self.cache = np.array(self.cache)
        return out+b

    def backward(self, dout):
        x = self.x
        W, b = self.params
        dW, db = self.grads

        dW_tmp = np.zeros_like(dW)
        dx = np.zeros_like(x)
        
        for i in range(self.batch_size):
            r = 0
            for j in range(dout.shape[1]):
                dx[i][r : r+W.shape[0]] += W * dout[i][j]
                dW_tmp += self.cache[j][i] * dout[i][j]
                r += self.stride

        self.grads[0][...] = dW_tmp
        self.grads[1][...] = np.sum(dout)

        return dx

class BatchPooling:
    def __init__(self):
        self.params, self.grads = [], []
        self.x = None
        self.argmax = None

    def forward(self, x):
        self.x = x
        self.argmax = np.argmax(x, axis=1)
        out = self.x.max(axis=1)
        return out

    def backward(self, dout):
        dx = np.zeros_like(self.x)
        for i in range(dx.shape[0]):
            dx[i, self.argmax[i]] = dout.flatten()[i]
        return dx

class BatchEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = np.empty((self.idx.shape[0], self.idx.shape[1], 300))
        for i in range(self.idx.shape[0]):
            out[i] = W[self.idx[i]]
        return out

    def backward(self, dout):
        W, = self.params
        dW = np.zeros_like(W).astype('f')
        for i in range(self.idx.shape[0]):
            for j, idx in enumerate(self.idx[i]):
                dW[idx] += dout[i][j]
        self.grads[0][...] = dW
        return None