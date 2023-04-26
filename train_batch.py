import pickle, sys
import random
import numpy as np
from tqdm.auto import tqdm
from batch_models import *
# from utils import *
from optimizers import *
np.random.seed(0)

def make_vec(data):
    xs, ts = [], []
    for d in data:
        leng = len(d['text'].split())
        tmp = [word_idx_map[w] for w in d["text"].split()] + [0]*(max_l-leng)
        xs.append(tmp)
        ts.append(d['y'])
        
    return np.array(xs), np.array(ts)

def accur(data):
    acc = 0
    cnt = 0
    xs, ts = make_vec(data)
    max_iter = len(xs) // 50
    for i in tqdm(range(max_iter), desc="calculating acc"):
        batch_x = xs[i*50 : (i+1)*50]
        batch_t = ts[i*50 : (i+1)*50]
        prediction = model.predict(batch_x)
        for i in range(len(prediction)):
            cnt += 1
            if prediction[i] == batch_t[i]:
                acc +=1

    return acc/cnt

def train(model, data):
    dev_acc = []
    dev_loss = 1000000000
    epoch = 100
    batch_size = 50
    optimizer = AdaDelta()
    train_data, dev_data, test_data = [], [], []
    for d in data:
        if d['split'] == 0:
            test_data.append(d)
        elif d['split'] == 1:
            dev_data.append(d)
        else:
            train_data.append(d)
    
    random.shuffle(train_data)
    xs, ts = make_vec(train_data)
    dev_xs, dev_ts = make_vec(dev_data)
    max_iter = len(train_data) // batch_size

    for e in tqdm(range(epoch)):
        print("Epoch: " +str(e+1))
        loss = 0
        cnt = 0
        for i in tqdm(range(max_iter)):
            cnt += 1
            batch_x = xs[i*batch_size : (i+1)*batch_size]
            batch_t = ts[i*batch_size : (i+1)*batch_size]        
            l = model.forward(batch_x, batch_t)
            loss += l
            model.backward()
            optimizer.update(model.params, model.grads)
            if cnt % 6 == 0:
                print("loss : ", loss / cnt)
                loss, cnt = 0, 0
        dev_tmp = accur(dev_data)
        print("Dev set acc after epoch : " + str(e+1))
        print(dev_tmp)
        dev_loss_tmp = 0
        for i in range(len(dev_data)//batch_size):
            dev_loss_tmp += model.forward(dev_xs[i*batch_size : (i+1)*batch_size], dev_ts[i*batch_size : (i+1)*batch_size])
        print("Dev set loss after epoch : " + str(e+1))
        print(dev_loss_tmp)
        if (dev_loss_tmp > dev_loss):
            print("early stopping at epoch : " + str(e+1))
            break
        dev_loss = dev_loss_tmp

        # if len(dev_acc) >= 3:
        #     if dev_tmp < np.mean(dev_acc):
        #         if e > 10:
        #             print("early stopping at: " + str(e+1))
        #             break
        #         else:
        #             dev_acc.pop(0)
        #             dev_acc.append(dev_tmp)
        #     else:
        #         dev_acc.pop(0)
        #         dev_acc.append(dev_tmp)
        # elif len(dev_acc) < 3:
        #     dev_acc.append(dev_tmp)

    print("Final acc")
    print(accur(test_data))

if __name__ == "__main__":
    model = sys.argv[1]
    data = sys.argv[2]
    conv_w = np.random.randint(3, 6, 100)
    if model == "rand":
        if data == "MR":
            with open('mr.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W2, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                            max_l=max_l, weight_decay_lambda=0.1)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/MR_rand_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-rand model saved!")

        elif data == "CR":
            with open('cr.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W2, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                            max_l=max_l, weight_decay_lambda=0.1)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/CR_rand_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-rand model saved!")

        elif data == "Subj":
            with open('subj.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W2, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                            max_l=max_l, weight_decay_lambda=0.1)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/Subj_rand_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-rand model saved!")

        elif data == "MPQA":
            with open('mpqa.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W2, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                            max_l=max_l, weight_decay_lambda=0.1)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/MPQA_rand_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-rand model saved!")

        elif data == "TREC":
            with open('trec.bin', 'rb') as f:
                train_data, test_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W2, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=6,
                          max_l=max_l, weight_decay_lambda=0.1)
            train_TREC(model, train_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/TREC_rand_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-rand model saved!")
        
        elif data == "SST1":
            with open('sst1.bin', 'rb') as f:
                train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W2, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=5,
                          max_l=max_l, weight_decay_lambda=0.1)
            train_SST(model, train_data, dev_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/SST1_rand_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-rand model saved!")

        elif data == "SST2":
            with open('sst2.bin', 'rb') as f:
                train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W2, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, weight_decay_lambda=0.1)
            train_SST(model, train_data, dev_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/SST2_rand_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-rand model saved!")

    elif model == "static":
        if data == "MR":
            with open('mr.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4TStatic(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, weight_decay_lambda=0.1)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/MR_static_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-static model saved!")
        
        elif data == "CR":
            with open('cr.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4TStatic(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, weight_decay_lambda=0.1)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/CR_static_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-static model saved!")

        elif data == "MPQA":
            with open('mpqa.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4TStatic(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, weight_decay_lambda=0.1)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/MPQA_static_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-static model saved!")

        elif data == "Subj":
            with open('subj.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4TStatic(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, weight_decay_lambda=0.1)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/Subj_static_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-static model saved!")

        elif data == "TREC":
            with open('trec.bin', 'rb') as f:
                train_data, test_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4TStatic(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=6,
                          max_l=max_l, weight_decay_lambda=0.1)
            train_TREC(model, train_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/TREC_static_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-static model saved!")
        
        elif data == "SST1":
            with open('sst1.bin', 'rb') as f:
                train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4TStatic(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=5,
                          max_l=max_l, weight_decay_lambda=0.1)
            train_SST(model, train_data, dev_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/SST1_static_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-static model saved!")

        elif data == "SST2":
            with open('sst2.bin', 'rb') as f:
                train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4TStatic(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, weight_decay_lambda=0.1)
            train_SST(model, train_data, dev_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/SST2_static_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-static model saved!")

    elif model == "non-static":
        if data == "MR":
            with open('mr.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, weight_decay_lambda=0.1)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/MR_nonstatic_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-nonstatic model saved!")

        elif data == "CR":
            with open('cr.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, weight_decay_lambda=0.1)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/CR_nonstatic_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-nonstatic model saved!")

        elif data == "MPQA":
            with open('mpqa.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, weight_decay_lambda=0.1)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/MPQA_nonstatic_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-nonstatic model saved!")

        elif data == "Subj":
            with open('subj.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, weight_decay_lambda=0.1)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/Subj_nonstatic_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-nonstatic model saved!")

        elif data == "TREC":
            with open('trec.bin', 'rb') as f:
                train_data, test_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=6,
                          max_l=max_l, weight_decay_lambda=0.1)
            train_TREC(model, train_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/TREC_non-static_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-non-static model saved!")
        
        elif data == "SST1":
            with open('sst1.bin', 'rb') as f:
                train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=5,
                          max_l=max_l, weight_decay_lambda=0.1)
            train_SST(model, train_data, dev_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/SST1_non-static_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-non-static model saved!")

        elif data == "SST2":
            with open('sst2.bin', 'rb') as f:
                train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, weight_decay_lambda=0.1)
            train_SST(model, train_data, dev_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/SST2_non-static_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-non-static model saved!")

    elif model == "multichannel":
        if data == "MR":
            with open('mr.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = MultiCNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, weight_decay_lambda=0.1)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/MR_multichannel_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-multichannel model saved!")

        elif data == "CR":
            with open('cr.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = MultiCNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, weight_decay_lambda=0.1)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/CR_multichannel_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-multichannel model saved!")

        elif data == "MPQA":
            with open('mpqa.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = MultiCNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, weight_decay_lambda=0.1)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/MPQA_multichannel_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-multichannel model saved!")

        elif data == "Subj":
            with open('subj.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = MultiCNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, weight_decay_lambda=0.1)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/Subj_multichannel_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-multichannel model saved!")

        elif data == "TREC":
            with open('trec.bin', 'rb') as f:
                train_data, test_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = MultiCNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=6,
                          max_l=max_l, weight_decay_lambda=0.1)
            train_TREC(model, train_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/TREC_multichannel_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-multichannel model saved!")
        
        elif data == "SST1":
            with open('sst1.bin', 'rb') as f:
                train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = MultiCNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=5,
                          max_l=max_l, weight_decay_lambda=0.1)
            train_SST(model, train_data, dev_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/SST1_multichannel_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-multichannel model saved!")

        elif data == "SST2":
            with open('sst2.bin', 'rb') as f:
                train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = MultiCNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, weight_decay_lambda=0.1)
            train_SST(model, train_data, dev_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/SST2_multichannel_batch.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-multichannel model saved!")