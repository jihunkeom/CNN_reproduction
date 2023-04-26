import numpy as np
from layers import *
np.random.seed(0)

class FC:
    def __init__(self, input_size, hidden_size, output_size, dropout, l2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.l2 = l2

        W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)
        b1 = np.zeros(self.hidden_size)
        b2 = np.zeros(self.output_size)

        self.layers = []
        self.layers.append(Affine(W1, b1))
        self.layers.append(Relu())
        self.layers.append(Dropout(dropout))
        self.layers.append(Affine(W2, b2))

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        self.loss_layer = SoftmaxWithLoss()

    def predict(self, x, train=False):
        if self.l2 == None:
            if train:
                for layer in self.layers:
                    if layer.name == "Dropout":
                        x = layer.forward(x, True)
                    else:
                        x = layer.forward(x)
            else:
                for layer in self.layers:
                    x = layer.forward(x)

        else:
            if train:
                for layer in self.layers:
                    if layer.name == "Dropout":
                        x = layer.forward(x, True)
                    else:
                        if layer.name == "Affine":
                            for param in layer.params:
                                norm = np.linalg.norm(param)
                                if norm > self.l2:
                                    param *= (self.l2/norm)
                        x = layer.forward(x)
            else:
                for layer in self.layers:
                    x = layer.forward(x)
        return x

    def loss(self, x, t):
        score = self.predict(x, True)
        return self.loss_layer.forward(score, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = np.sum(y==t) / float(x.shape[0])
        return acc

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout

class Conv:
    def __init__(self, conv_W, conv_b, stride):
        conv_filter = (np.random.rand(conv_W, 300)).astype('f') * 0.01
        self.layers = [
            Convolution(conv_filter, conv_b, stride),
            Relu(),
            Pooling(),
        ]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout

class BatchConv:
    def __init__(self, conv_W, conv_b, stride):
        conv_filter = (np.random.rand(conv_W, 300)).astype('f') * 0.01
        self.layers = [
            BatchConvolution(conv_filter, conv_b, stride),
            Relu(),
            BatchPooling(),
        ]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout

class BatchFC:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)
        b1 = np.zeros(self.hidden_size)
        b2 = np.zeros(self.output_size)

        self.layers = []
        self.layers.append(Affine(W1, b1))
        self.layers.append(Relu())
        self.layers.append(Dropout())
        self.layers.append(Affine(W2, b2))

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        self.loss_layer = SoftmaxWithLoss()

    def predict(self, x, train=False):
        if train:
            for layer in self.layers:
                if layer.name == "Dropout":
                    x = layer.forward(x, True)
                else:
                    if layer.name == "Affine":
                        for param in layer.params:
                            norm = np.linalg.norm(param)
                            if norm > 3:
                                param *= (3/norm)
                    x = layer.forward(x)
        else:
            for layer in self.layers:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        score = self.predict(x, True)
        return self.loss_layer.forward(score, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = np.sum(y==t) / float(x.shape[0])
        return acc

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout