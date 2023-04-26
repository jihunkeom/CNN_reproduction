import pickle, sys
import numpy as np
from tqdm.auto import tqdm


class AdaDelta:
    def __init__(self, gamma=0.95):
        self.gamma = gamma
        self.E_g = None
        self.E_p = None
        self.delta_p = None

    def update(self, params, grads):
        if self.E_g == None:
            self.E_g = []
            self.E_p = []
            self.delta_p = []
            for val in params:
                self.E_g.append(np.zeros_like(val))
                self.E_p.append(np.zeros_like(val))
                self.delta_p.append(np.zeros_like(val))

        for i in range(len(params)):
            self.E_g[i] = self.gamma*self.E_g[i] + (1-self.gamma)*(grads[i]*grads[i])
            self.delta_p[i] = -np.sqrt(self.E_p[i]+1e-7)/np.sqrt(self.E_g[i]+1e-7) * grads[i]
            self.E_p[i] = self.gamma*self.E_p[i] + (1-self.gamma)*(self.delta_p[i]**2)
            params[i] += self.delta_p[i]

        return params

model = sys.argv[1]
data = sys.argv[2]

if data == "SST1":
    with open('sst1.bin', 'rb') as f:
        train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
elif data == "SST2":
    with open('sst2.bin', 'rb') as f:
        train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)

def make_vec(data):
    xs, ts = [], []
    for d in data:
        leng = len(d['text'].split())
        tmp = [word_idx_map[w] for w in d["text"].split()] + [0]*(max_l-leng)
        ts.append(d['y'])
        xs.append(tmp)
    return np.array(xs), np.array(ts)

train_data = train_data + dev_data
xs, ts = make_vec(train_data)

conv_w = np.random.randint(3, 6, 100)
if data == "SST1":
    embed_matrix = W2
    embed_grad = np.zeros_like(embed_matrix)
    conv_filters = [0.01*np.random.randn(conv_w[i], 300) for i in range(100)]
    conv_grads = [np.zeros_like(conv_filters[i]) for i in range(100)]
    conv_b = [np.zeros(1) for _ in range(100)]
    conv_b_grads = [np.zeros_like(conv_b[i]) for i in range(100)]
    affine_W1 = np.random.randn(100, 100) * np.sqrt(2.0 / 100)
    affine_W2 = np.random.randn(100, 5) * np.sqrt(2.0 / 100)
    affine_b1 = np.zeros(100)
    affine_b2 = np.zeros(5)
    dW1 = np.zeros_like(affine_W1)
    dW2 = np.zeros_like(affine_W2)
    db1 = np.zeros_like(affine_b1)
    db2 = np.zeros_like(affine_b2)

elif data=="SST2":
    embed_matrix = W2
    embed_grad = np.zeros_like(embed_matrix)
    conv_filters = [0.01*np.random.randn(conv_w[i], 300) for i in range(100)]
    conv_grads = [np.zeros_like(conv_filters[i]) for i in range(100)]
    conv_b = [np.zeros(1) for _ in range(100)]
    conv_b_grads = [np.zeros_like(conv_b[i]) for i in range(100)]
    affine_W1 = np.random.randn(100, 100) * np.sqrt(2.0 / 100)
    affine_W2 = np.random.randn(100, 2) * np.sqrt(2.0 / 100)
    affine_b1 = np.zeros(100)
    affine_b2 = np.zeros(2)
    dW1 = np.zeros_like(affine_W1)
    dW2 = np.zeros_like(affine_W2)
    db1 = np.zeros_like(affine_b1)
    db2 = np.zeros_like(affine_b2)


optimizer = AdaDelta()

def accur(data):
    acc = 0
    xs, ts = make_vec(data)
    for i in tqdm(range(len(ts)), desc="calculating acc"):
        input_x, t = xs[i], ts[i]
        x = embed_matrix[input_x]
        penultimate_layer = []
        arg_max = []
        pool_inputs = []
        conv_relu_masks = []
        caches = []
        for w in range(100):
            W = conv_filters[w]
            out_h = int(x.shape[0] - W.shape[0]) + 1
            out_w = 1
            out = np.empty((out_h, out_w), dtype="f")

            r = 0
            cache = []
            for i in range(out_h):
                tmp = x[r: r + W.shape[0]]
                out[i] = np.sum(tmp * W)
                cache.append(tmp)
                r += 1
            caches.append(cache)
            out += conv_b[w]
            mask = (out <= 0)
            conv_relu_masks.append(mask)
            out[mask] = 0
            pool_inputs.append(out)
            arg_max.append(np.argmax(out))
            out = out.max()
            penultimate_layer.append(out)

        penultimate_layer = np.array(penultimate_layer).reshape(1, -1)
        affine_x1 = penultimate_layer
        out = np.dot(penultimate_layer, affine_W1) + affine_b1
        relu_mask = (out <= 0)
        out[relu_mask] = 0
        dropout_mask = np.random.rand(*out.shape) > 0.5
        out *= dropout_mask
        affine2_x = out
        out = np.dot(out, affine_W2) + affine_b2
        out = out.flatten()
        out -= np.max(out)
        y = np.exp(out) / np.sum(np.exp(out))
        y = y.reshape(1, -1)

        prediction = y.argmax()


        if prediction == ts[i]:
            acc += 1

        # print(prediction, ts[i])

    return acc/len(ts)

accuracies = []
accuracies.append(accur(test_data))
print(accuracies)

for epoch in tqdm(range(25)):
    loss = 0
    perm = np.random.permutation(len(xs))
    cnt = 0
    for iter in tqdm(perm):
        cnt += 1
        input_x, t = xs[iter], ts[iter]
        x = embed_matrix[input_x]
        penultimate_layer = []
        arg_max = []
        pool_inputs = []
        conv_relu_masks = []
        caches = []
        for w in range(100):
            W = conv_filters[w]
            out_h = int(x.shape[0] - W.shape[0]) + 1
            out_w = 1
            out = np.empty((out_h, out_w), dtype="f")

            r = 0
            cache = []
            for i in range(out_h):
                tmp = x[r: r + W.shape[0]]
                out[i] = np.sum(tmp * W)
                cache.append(tmp)
                r += 1
            caches.append(cache)
            out += conv_b[w]
            mask = (out <= 0)
            conv_relu_masks.append(mask)
            out[mask] = 0
            pool_inputs.append(out)
            arg_max.append(np.argmax(out))
            out = out.max()
            penultimate_layer.append(out)

        penultimate_layer = np.array(penultimate_layer).reshape(1, -1)
        affine_x1 = penultimate_layer
        out = np.dot(penultimate_layer, affine_W1) + affine_b1
        relu_mask = (out <= 0)
        out[relu_mask] = 0
        dropout_mask = np.random.rand(*out.shape) > 0.5
        out *= dropout_mask
        affine2_x = out
        out = np.dot(out, affine_W2) + affine_b2
        out = out.flatten()
        out -= np.max(out)
        y = np.exp(out) / np.sum(np.exp(out))
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)
        batch_size = y.shape[0]
        loss += -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7))
        
        if (cnt % 100 == 0) and (cnt > 0):
            print(loss)
            loss = 0

        #backward
        dout = y.copy()
        dout[np.arange(batch_size), t] -= 1
        dout /= batch_size
        db2 = np.sum(dout, axis=0)
        dW2 = np.dot(affine2_x.T, dout)
        dout = np.dot(dout, affine_W2.T)
        dout *= dropout_mask
        dout[relu_mask] = 0
        db1 = np.sum(dout, axis=0)
        dW1 = np.dot(affine_x1.T, dout)
        tmp = np.dot(dout, affine_W1.T)
        tmp = tmp.flatten()


        dout = (np.zeros_like(pool_inputs[0])).astype('f')
        dout[arg_max[0]] = tmp[0]
        dout[conv_relu_masks[0]] = 0
        dW_tmp = np.zeros_like(conv_filters[0])
        dx = np.zeros_like(x)
        r = 0
        for i in range(dout.shape[0]):
            dx[r: r + conv_filters[0].shape[0]] = conv_filters[0] * dout[i]
            dW_tmp += dout[i] * caches[0][i]
            r += 1

        conv_grads[0][...] = dW_tmp
        conv_b_grads[0][...] = np.sum(dout)

        for j in range(1, 100):
            dout = (np.zeros_like(pool_inputs[j])).astype('f')
            dout[arg_max[j]] = tmp[j]
            dout[conv_relu_masks[j]] = 0
            dW_tmp = np.zeros_like(conv_filters[j])
            dx_tmp = np.zeros_like(x)
            r = 0
            for i in range(dout.shape[0]):
                dx_tmp[r: r + conv_filters[j].shape[0]] = conv_filters[j] * dout[i]
                dW_tmp += dout[i] * caches[j][i]
                r += 1
            dx += dx_tmp
            conv_grads[j][...] = dW_tmp
            conv_b_grads[j][...] = np.sum(dout)

        embed_grad[...] = 0
        np.add.at(embed_grad, input_x, dx)

        params = [embed_matrix] + conv_filters + conv_b + [affine_W1, affine_b1, affine_W2, affine_b2]
        grads = [embed_grad] + conv_grads + conv_b_grads + [dW1, db1, dW2, db2]
        params = optimizer.update(params, grads)
        embed_matrix, conv_filters, conv_b, affine_W1, affine_b1, affine_W2, affine_b2 = params[0], params[1:101], params[101:201], params[201], params[202], params[203], params[204]


accuracies.append(accur(test_data))
print(accuracies)