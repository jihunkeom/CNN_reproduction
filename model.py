import numpy as np
from partials import *
from layers import *
np.random.seed(0)

class CNN4T:
    def __init__(self, embed_w, conv_w, conv_b, stride, num_filters, hidden_size, output_size, max_l, dropout, l2):
        self.num_filters = num_filters
        self.max_l = max_l
        self.embedding_layer = Embedding(embed_w)
        self.conv_layers = [Conv(conv_w[i], conv_b, stride) for i in range(num_filters)]
        self.FC = FC(num_filters, hidden_size, output_size, dropout, l2)
        self.penultimate_layer = None

        self.params, self.grads = [], []
        self.params += self.embedding_layer.params
        self.grads += self.embedding_layer.grads
        for layer in self.conv_layers:
            self.params += layer.params
            self.grads += layer.grads
        self.params += self.FC.params
        self.grads += self.FC.grads

    def pad(self, x):
        if x.shape[0] < self.max_l:
            tmp = np.zeros((self.max_l - x.shape[0], 300), dtype='f')
            x = np.vstack([x, tmp])
        return x

    def predict(self, x):
        self.penultimate_layer = []
        x = self.embedding_layer.forward(x)
        # x = self.pad(x)
        for layer in self.conv_layers:
            self.penultimate_layer.append(layer.forward(x))
        self.penultimate_layer = np.array(self.penultimate_layer).reshape(1, -1)
        out = self.FC.predict(self.penultimate_layer)
        return np.argmax(out.flatten())

    def forward(self, x, t):
        self.penultimate_layer = []
        x = self.embedding_layer.forward(x)
        # x = self.pad(x)
        for layer in self.conv_layers:
            self.penultimate_layer.append(layer.forward(x))
        self.penultimate_layer = np.array(self.penultimate_layer).reshape(1, -1)
        loss = self.FC.loss(self.penultimate_layer, t)
        return loss

    def backward(self):
        tmp = self.FC.backward(1)
        tmp = tmp.flatten()
        dout = self.conv_layers[0].backward(tmp[0])
        i = 1
        for layer in self.conv_layers[1:]:
            dout += layer.backward(tmp[i])
            i += 1
        self.embedding_layer.backward(dout)

    def accuarcy(self, x, t):
        self.penultimate_layer = []
        x = self.embedding_layer.forward(x)
        x = self.pad(x)
        for layer in self.conv_layers:
            self.penultimate_layer.append(layer.forward(x))
        self.penultimate_layer = np.array(self.penultimate_layer).reshape(1, -1)
        acc = self.FC.accuracy(self.penultimate_layer, t)
        return acc

class CNN4TStatic:
    def __init__(self, embed_w, conv_w, conv_b, stride, num_filters, hidden_size, output_size, max_l, dropout, l2):
        self.num_filters = num_filters
        self.max_l = max_l
        self.embedding_layer = Embedding(embed_w)
        self.conv_layers = [Conv(conv_w[i], conv_b, stride) for i in range(num_filters)]
        self.FC = FC(self.num_filters, hidden_size, output_size, dropout, l2)
        self.penultimate_layer = None

        self.params, self.grads = [], []
        self.params += self.embedding_layer.params
        self.grads += self.embedding_layer.grads
        for layer in self.conv_layers:
            self.params += layer.params
            self.grads += layer.grads
        self.params += self.FC.params
        self.grads += self.FC.grads

    def pad(self, x):
        if x.shape[0] < self.max_l:
            tmp = np.zeros((self.max_l - x.shape[0], 300), dtype='f')
            x = np.vstack([x, tmp])
        return x

    def predict(self, x):
        self.penultimate_layer = []
        x = self.embedding_layer.forward(x)
        # x = self.pad(x)
        for layer in self.conv_layers:
            self.penultimate_layer.append(layer.forward(x))
        self.penultimate_layer = np.array(self.penultimate_layer).reshape(1, -1)
        out = self.FC.predict(self.penultimate_layer)
        return np.argmax(out.flatten())

    def forward(self, x, t):
        self.penultimate_layer = []
        x = self.embedding_layer.forward(x)
        # x = self.pad(x)
        for layer in self.conv_layers:
            self.penultimate_layer.append(layer.forward(x))
        self.penultimate_layer = np.array(self.penultimate_layer).reshape(1, -1)
        loss = self.FC.loss(self.penultimate_layer, t)
        return loss

    def backward(self):
        tmp = self.FC.backward(1)
        tmp = tmp.flatten()
        dout = self.conv_layers[0].backward(tmp[0])
        i = 1
        for layer in self.conv_layers[1:]:
            dout += layer.backward(tmp[i])
            i += 1
        # self.embedding_layer.backward(dout)

    def accuarcy(self, x, t):
        self.penultimate_layer = []
        x = self.embedding_layer.forward(x)
        x = self.pad(x)
        for layer in self.conv_layers:
            self.penultimate_layer.append(layer.forward(x))
        self.penultimate_layer = np.array(self.penultimate_layer).reshape(1, -1)
        acc = self.FC.accuracy(self.penultimate_layer, t)
        return acc

class MultiCNN4T:
    def __init__(self, embed_w, conv_w, conv_b, stride, num_filters, hidden_size, output_size, max_l, dropout, l2):
        self.num_filters = num_filters
        self.max_l = max_l
        self.static_embedding_layer = Embedding(embed_w)
        self.embedding_layer = Embedding(embed_w)
        self.conv_layers = [Conv(conv_w[i], conv_b, stride) for i in range(num_filters)]
        self.FC = FC(self.num_filters, hidden_size, output_size, dropout, l2)
        self.penultimate_layer = None

        self.params, self.grads = [], []
        self.params += self.static_embedding_layer.params
        self.grads += self.static_embedding_layer.grads
        self.params += self.embedding_layer.params
        self.grads += self.embedding_layer.grads
        for layer in self.conv_layers:
            self.params += layer.params
            self.grads += layer.grads
        self.params += self.FC.params
        self.grads += self.FC.grads

    def pad(self, x):
        if x.shape[0] < self.max_l:
            tmp = np.zeros((self.max_l - x.shape[0], 300), dtype='f')
            x = np.vstack([x, tmp])
        return x

    def predict(self, x):
        self.penultimate_layer = []
        x1 = self.static_embedding_layer.forward(x)
        x1 = self.pad(x1)
        x2 = self.embedding_layer.forward(x)
        x2 = self.pad(x2)
        for layer in self.conv_layers:
            tmp = layer.forward(x1)
            tmp += layer.forward(x2)
            self.penultimate_layer.append(tmp)
        self.penultimate_layer = np.array(self.penultimate_layer).reshape(1, -1)
        out = self.FC.predict(self.penultimate_layer)
        return np.argmax(out.flatten())

    def forward(self, x, t):
        self.penultimate_layer = []
        x1 = self.static_embedding_layer.forward(x)
        x1 = self.pad(x1)
        x2 = self.embedding_layer.forward(x)
        x2 = self.pad(x2)
        for layer in self.conv_layers:
            tmp = layer.forward(x1)
            tmp += layer.forward(x2)
            self.penultimate_layer.append(tmp)
        self.penultimate_layer = np.array(self.penultimate_layer).reshape(1, -1)
        loss = self.FC.loss(self.penultimate_layer, t)
        return loss

    def backward(self):
        tmp = self.FC.backward(1)
        tmp = tmp.flatten()
        dout = self.conv_layers[0].backward(tmp[0])
        i = 1
        for layer in self.conv_layers[1:]:
            dout += layer.backward(tmp[i])
            i += 1

        self.embedding_layer.backward(dout)
