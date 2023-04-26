import pickle, sys
import numpy as np
from models import *
from utils import *
np.random.seed(0)

with open('./models/CR_rand.pkl', 'rb') as f:
    params = pickle.load(f)

params = [p.astype('f') for p in params]

model2 = CNN4T(embed_w=W2, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                max_l=max_l, weight_decay_lambda=0.1)

for i, param in enumerate(model2.params):
    try:
        param[...] = params[i]
    except:
        param = params[i]

test_data = []
for d in data:
    if d['split'] == 0:
        test_data.append(d)

acc = 0
xs, ts = make_vec(test_data)
for i in tqdm(range(len(ts)), desc="calculating acc"):
    prediction = model2.predict(xs[i])
    if prediction == ts[i]:
        acc += 1

print(acc / len(ts))