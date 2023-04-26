import pickle, sys
import numpy as np
from tqdm.auto import tqdm
from model import *
from optimizers import *
np.random.seed(0)

def make_vec(data):
    xs, ts = [], []
    for d in data:
        leng = len(d['text'].split())
        tmp = [word_idx_map[w] for w in d["text"].split()] + [0]*(max_l-leng)
        ts.append(d['y'])
        xs.append(tmp)
    return np.array(xs), np.array(ts)

def accur(data):
    acc = 0
    xs, ts = make_vec(data)
    for i in tqdm(range(len(ts)), desc="calculating acc"):
        prediction = model.predict(xs[i])
        if prediction == ts[i]:
            acc += 1

    return acc/len(ts)

def train(model, data):
## ES on loss
    # dev_loss = 0
    epoch = 45
    optimizer = AdaDelta()
    train_data, test_data = [], []
    for d in data:
        if d['split'] == 0:
            test_data.append(d)
        else:
            train_data.append(d)
    xs, ts = make_vec(train_data)
    # dev_xs, dev_ts = make_vec(dev_data)

    # for i in tqdm(range(len(dev_ts)), desc="calculating initial dev loss"):
    #     dev_loss += model.forward(dev_xs[i], dev_ts[i])
    # print("initial dev loss : ", dev_loss)
    
    for e in tqdm(range(epoch)):
        print("Epoch: " +str(e+1))
        loss = 0
        cnt = 0
        perm = np.random.permutation(len(xs))
        for i in tqdm(perm):
            cnt += 1
            l = model.forward(xs[i], ts[i])
            loss += l
            model.backward()
            optimizer.update(model.params, model.grads)

            if cnt%300 == 0:
                print("loss: ", loss/cnt)
                loss, cnt = 0, 0

        # dev_acc = accur(dev_data)
        # dev_loss_tmp = 0
        # for i in tqdm(range(len(dev_ts)), desc="calculating dev loss"):
            # dev_loss_tmp += model.forward(dev_xs[i], dev_ts[i])
        # print("Dev set acc after epoch : " + str(e+1))
        # print(dev_acc)
        # print("Dev set loss after epoch : " + str(e+1))
        # print(dev_loss_tmp)
        # if (dev_loss_tmp > dev_loss):
            # print("early stopping at epoch : " + str(e+1))
            # break
        # dev_loss = dev_loss_tmp

    print("Final acc")
    print(accur(test_data))


# def train(model, data):
# #ES on acc
#     dev_acc = []
#     # dev_acc = 0
#     epoch = 100
#     optimizer = AdaDelta()
#     train_data, dev_data, test_data = [], [], []
#     for d in data:
#         if d['split'] == 0:
#             test_data.append(d)
#         elif d['split'] == 1:
#             dev_data.append(d)
#         else:
#             train_data.append(d)
#     xs, ts = make_vec(train_data)

#     for e in tqdm(range(epoch)):
#         print("Epoch: " +str(e+1))
#         loss = 0
#         cnt = 0
#         perm = np.random.permutation(len(xs))
#         for i in tqdm(perm):
#             cnt += 1
#             l = model.forward(xs[i], ts[i])
#             loss += l
#             model.backward()
#             optimizer.update(model.params, model.grads)

#             if cnt%300 == 0:
#                 print("loss: ", loss/cnt)
#                 loss, cnt = 0, 0

#         dev_tmp = accur(dev_data)
#         print("Dev set acc after epoch : " + str(e+1))
#         print(dev_tmp)
#         # if dev_tmp < dev_acc :
#         #     print("early stopping at: " + str(e+1))
#         #     break
#         # else:
#         #     dev_acc = dev_tmp
        
#         if len(dev_acc) >= 2:
#             if dev_tmp < np.mean(dev_acc):
#                 print("early stopping at: " + str(e+1))
#                 break
#             else:
#                 dev_acc.pop(0)
#                 dev_acc.append(dev_tmp)
#         elif len(dev_acc) < 2:
#             dev_acc.append(dev_tmp)

#     print("Final acc")
#     print(accur(test_data))

def train_TREC(model, train_data, test_data):
    # dev_loss = 0
    epoch = 50
    optimizer = AdaDelta()
    # train_data, dev_data = [], []
    # for d in train_dev_data:
    #     if d['split'] == 0:
    #         dev_data.append(d)
    #     else:
    #         train_data.append(d)
    xs, ts = make_vec(train_data)

    for e in tqdm(range(epoch)):
        print("Epoch: " +str(e+1))
        loss = 0
        cnt = 0
        perm = np.random.permutation(len(xs))
        for i in tqdm(perm):
            cnt += 1
            l = model.forward(xs[i], ts[i])
            loss += l
            model.backward()
            optimizer.update(model.params, model.grads)

            if cnt%300 == 0:
                print("loss: ", loss/cnt)
                loss, cnt = 0, 0

        # dev_acc = accur(dev_data)
        # dev_loss_tmp = 0
        # for i in tqdm(range(len(dev_ts)), desc="calculating dev loss"):
        #     dev_loss_tmp += model.forward(dev_xs[i], dev_ts[i])
        # print("Dev set acc after epoch : " + str(e+1))
        # print(dev_acc)
        # print("Dev set loss after epoch : " + str(e+1))
        # print(dev_loss_tmp)
        # if (dev_loss_tmp > dev_loss):
        #     print("early stopping at epoch : " + str(e+1))
        #     break
        # dev_loss = dev_loss_tmp

    print("Final acc")
    print(accur(test_data))

def train_SST(model, train_data, dev_data, test_data):
    dev_loss = 0
    epoch = 30
    optimizer = AdaDelta()
    train_data = train_data + dev_data
    xs, ts = make_vec(train_data)

    for e in tqdm(range(epoch)):
        print("Epoch: " +str(e+1))
        loss = 0
        cnt = 0
        perm = np.random.permutation(len(xs))
        for i in tqdm(perm):
            cnt += 1
            l = model.forward(xs[i], ts[i])
            loss += l
            model.backward()
            optimizer.update(model.params, model.grads)

            if cnt%300 == 0:
                print("loss: ", loss/cnt)
                loss, cnt = 0, 0

        # dev_acc = accur(dev_data)
        # dev_loss_tmp = 0
        # for i in tqdm(range(len(dev_ts)), desc="calculating dev loss"):
        #     dev_loss_tmp += model.forward(dev_xs[i], dev_ts[i])
        # print("Dev set acc after epoch : " + str(e+1))
        # print(dev_acc)
        # print("Dev set loss after epoch : " + str(e+1))
        # print(dev_loss_tmp)
        # if (dev_loss_tmp > dev_loss):
        #     print("early stopping at epoch : " + str(e+1))
        #     break
        # dev_loss = dev_loss_tmp

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
                            max_l=max_l, dropout=0.5, l2=3)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/MR_rand.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-rand model saved!")

        elif data == "CR":
            with open('cr.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W2, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                            max_l=max_l, dropout=0.5, l2=3)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/CR_rand.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-rand model saved!")

        elif data == "Subj":
            with open('subj.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W2, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                            max_l=max_l, dropout=0.5, l2=3)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/Subj_rand.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-rand model saved!")

        elif data == "MPQA":
            with open('mpqa.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W2, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                            max_l=max_l, dropout=0.5, l2=3)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/MPQA_rand.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-rand model saved!")

        elif data == "TREC":
            with open('trec.bin', 'rb') as f:
                train_data, test_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W2, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=6,
                          max_l=max_l, dropout=0.5, l2=3)
            train_TREC(model, train_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/TREC_rand.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-rand model saved!")
        
        elif data == "SST1":
            with open('sst1.bin', 'rb') as f:
                train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W2, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=5,
                          max_l=max_l, dropout=0.5, l2=3)
            train_SST(model, train_data, dev_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/SST1_rand.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-rand model saved!")

        elif data == "SST2":
            with open('sst2.bin', 'rb') as f:
                train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W2, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, dropout=0.5, l2=3)
            train_SST(model, train_data, dev_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/SST2_rand.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-rand model saved!")

    elif model == "static":
        if data == "MR":
            with open('mr.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4TStatic(embed_w=W, conv_w=(np.ones(1000)*5).astype("int8"), conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, dropout=0.7, l2=4)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/MR_static.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-static model saved!")
        
        elif data == "CR":
            with open('cr.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4TStatic(embed_w=W, conv_w=(np.ones(600)*7).astype("int8"), conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, dropout=0.3, l2=4)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/CR_static.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-static model saved!")

        elif data == "MPQA":
            with open('mpqa.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4TStatic(embed_w=W, conv_w=(np.ones(600)*5).astype("int8"), conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, dropout=0.3, l2=None)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/MPQA_static.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-static model saved!")

        elif data == "Subj":
            with open('subj.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4TStatic(embed_w=W, conv_w=(np.ones(400)*3).astype("int8"), conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, dropout=0.3, l2=3)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/Subj_static.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-static model saved!")

        elif data == "TREC":
            with open('trec.bin', 'rb') as f:
                train_data, test_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4TStatic(embed_w=W, conv_w=(np.ones(200)*3).astype("int8"), conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=6,
                          max_l=max_l, dropout=0.1, l2=15)
            train_TREC(model, train_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/TREC_static.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-static model saved!")
        
        elif data == "SST1":
            with open('sst1.bin', 'rb') as f:
                train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4TStatic(embed_w=W, conv_w=(np.ones(50)*3).astype("int8"), conv_b=0, stride=1, num_filters=50, hidden_size=100, output_size=5,
                          max_l=max_l, dropout=0.7, l2=3)
            train_SST(model, train_data, dev_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/SST1_static.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-static model saved!")

        elif data == "SST2":
            with open('sst2.bin', 'rb') as f:
                train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4TStatic(embed_w=W, conv_w=(np.ones(600)*5).astype("int8"), conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, dropout=0.5, l2=30)
            train_SST(model, train_data, dev_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/SST2_static.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-static model saved!")

    elif model == "non-static":
        if data == "MR":
            with open('mr.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W, conv_w=(np.ones(400)*7).astype("int8"), conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, dropout=0.5, l2=4)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/MR_nonstatic.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-nonstatic model saved!")

        elif data == "CR":
            with open('cr.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W, conv_w=(np.ones(1000)*30).astype("int8"), conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, dropout=0., l2=None)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/CR_nonstatic.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-nonstatic model saved!")

        elif data == "MPQA":
            with open('mpqa.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W, conv_w=(np.ones(600)*7).astype("int8"), conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, dropout=0.1, l2=25)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/MPQA_nonstatic.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-nonstatic model saved!")

        elif data == "Subj":
            with open('subj.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W, conv_w=(np.ones(600)*7).astype("int8"), conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, dropout=0.3, l2=None)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/Subj_nonstatic.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-nonstatic model saved!")

        elif data == "TREC":
            with open('trec.bin', 'rb') as f:
                train_data, test_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W, conv_w=(np.ones(100)*3).astype("int8"), conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=6,
                          max_l=max_l, dropout=0.5, l2=15)
            train_TREC(model, train_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/TREC_non-static.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-non-static model saved!")
        
        elif data == "SST1":
            with open('sst1.bin', 'rb') as f:
                train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W, conv_w=(np.ones(100)*5).astype("int8"), conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=5,
                          max_l=max_l, dropout=0.7, l2=3)
            train_SST(model, train_data, dev_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/SST1_non-static.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-non-static model saved!")

        elif data == "SST2":
            with open('sst2.bin', 'rb') as f:
                train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = CNN4T(embed_w=W, conv_w=(np.ones(1000)*10).astype("int8"), conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, dropout=0.1, l2=30)
            train_SST(model, train_data, dev_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/SST2_non-static.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-non-static model saved!")

    elif model == "multichannel":
        if data == "MR":
            with open('mr.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = MultiCNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, dropout=0.5, l2=4)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/MR_multichannel.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-multichannel model saved!")

        elif data == "CR":
            with open('cr.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = MultiCNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, dropout=0.5, l2=4)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/CR_multichannel.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-multichannel model saved!")

        elif data == "MPQA":
            with open('mpqa.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = MultiCNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, dropout=0.5, l2=4)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/MPQA_multichannel.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-multichannel model saved!")

        elif data == "Subj":
            with open('subj.bin', 'rb') as f:
                data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = MultiCNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, dropout=0.5, l2=4)
            train(model, data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/Subj_multichannel.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-multichannel model saved!")

        elif data == "TREC":
            with open('trec.bin', 'rb') as f:
                train_data, test_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = MultiCNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=6,
                          max_l=max_l, dropout=0.5, l2=4)
            train_TREC(model, train_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/TREC_multichannel.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-multichannel model saved!")
        
        elif data == "SST1":
            with open('sst1.bin', 'rb') as f:
                train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = MultiCNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=5,
                          max_l=max_l, dropout=0.5, l2=4)
            train_SST(model, train_data, dev_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/SST1_multichannel.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-multichannel model saved!")

        elif data == "SST2":
            with open('sst2.bin', 'rb') as f:
                train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
            model = MultiCNN4T(embed_w=W, conv_w=conv_w, conv_b=0, stride=1, num_filters=100, hidden_size=100, output_size=2,
                          max_l=max_l, dropout=0.5, l2=4)
            train_SST(model, train_data, dev_data, test_data)
            params = [p.astype(np.float16) for p in model.params]
            with open("./models/SST2_multichannel.pkl", "wb") as f:
                pickle.dump(params, f)
            print("CNN-multichannel model saved!")