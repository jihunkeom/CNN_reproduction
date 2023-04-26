import sys, pickle, os
from collections import defaultdict
import pandas as pd
import numpy as np

def clean_str(string):
    string = string.strip().lower()
    string = string.replace(".", " .")
    string = string.replace(",", " ,")
    string = string.replace("!", " ! ")
    string = string.replace("?", " ? ")
    string = string.replace("\n", " ")
    string = string.replace("  ", " ")
    string = string.replace("\\", " ")
    string = string.replace("  n", "")
    string = string.replace("-", " ")
    string = string.replace(")", " )")
    string = string.replace("(", "( ")
    return string.strip().lower()

def read_MR_data(path_pos, path_neg, cv=10):
    data = []
    vocab = {}

    with open(path_pos, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = str(line)[2:-1]
            line = clean_str(line)
            words = set(line.split())
            for word in words:
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            datum = {"y": 1, "text": line, "num_words": len(line.split()), "split": np.random.randint(0, cv)}
            data.append(datum)

    with open(path_neg, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = str(line)[2:-1]
            line = clean_str(line)
            words = set(line.split())
            for word in words:
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            datum = {"y": 0, "text": line, "num_words": len(line.split()), "split": np.random.randint(0, cv)}
            data.append(datum)

    return data, vocab

def read_CR_data(path_pos, path_neg, cv=10):
    data = []
    vocab = {}

    with open(path_pos, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = str(line)[2:-1]
            line = clean_str(line)
            words = set(line.split())
            for word in words:
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            datum = {"y": 1, "text": line, "num_words": len(line.split()), "split": np.random.randint(0, cv)}
            data.append(datum)

    with open(path_neg, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = str(line)[2:-1]
            line = clean_str(line)
            words = set(line.split())
            for word in words:
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            datum = {"y": 0, "text": line, "num_words": len(line.split()), "split": np.random.randint(0, cv)}
            data.append(datum)

    return data, vocab

def read_MPQA_data(path_pos, path_neg, cv=10):
    data = []
    vocab = {}

    with open(path_pos, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = str(line)[2:-2]
            line = clean_str(line)
            words = set(line.split())
            for word in words:
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            datum = {"y": 1, "text": line, "num_words": len(line.split()), "split": np.random.randint(0, cv)}
            data.append(datum)

    with open(path_neg, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = str(line)[2:-2]
            line = clean_str(line)
            words = set(line.split())
            for word in words:
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            datum = {"y": 0, "text": line, "num_words": len(line.split()), "split": np.random.randint(0, cv)}
            data.append(datum)

    return data, vocab

def read_Subj_data(subj_path, obj_path, cv=10):
    data = []
    vocab = {}

    with open(subj_path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = str(line)[2:-1]
            line = clean_str(line)
            words = set(line.split())
            for word in words:
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            datum = {"y": 1, "text": line, "num_words": len(line.split()), "split": np.random.randint(0, cv)}
            data.append(datum)

    with open(obj_path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = str(line)[2:-1]
            line = clean_str(line)
            words = set(line.split())
            for word in words:
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            datum = {"y": 0, "text": line, "num_words": len(line.split()), "split": np.random.randint(0, cv)}
            data.append(datum)

    return data, vocab

def read_TREC_data(path):
    train_data = []
    test_data = []
    vocab = {}
    label_dict = {"NUM": 0, "HUM": 1, "ENTY":2, "DESC":3, "LOC":4, "ABBR":5}


    with open(path + "traindata.txt", "r", encoding='utf-8-sig') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = str(line)[:-2]
            line = line.replace(" :", " ")
            label, text = line.split(":")
            label = label.strip()
            text = text.lower()
            words = set(text.split())
            for word in words:
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            datum = {"y": label_dict[label], "text": text, "num_words": len(text.split()), "split": 1}
            train_data.append(datum)

    with open(path + "testdata.txt", "r", encoding='utf-8-sig') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = str(line)[:-2]
            line = line.replace(" :", " ")
            label, text = line.split(":")
            text = text.lower()
            words = set(text.split())
            for word in words:
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            datum = {"y": label_dict[label], "text": text, "num_words": len(text.split()), "split": 0}
            test_data.append(datum)

    return train_data, test_data, vocab

def read_SST1_data(path):
    train_data = []
    test_data = []
    dev_data = []
    vocab = {}

    with open(path + "stsa.fine.train.txt", "r", encoding='utf-8-sig') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = str(line)
            line = clean_str(line)
            label = line[0]
            text = line[2:]
            text = text.lower()
            words = set(text.split())
            for word in words:
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            datum = {"y": int(label), "text": text, "num_words": len(text.split()), "split": 1}
            train_data.append(datum)

    with open(path + "stsa.fine.phrases.train.txt", "r", encoding='utf-8-sig') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = str(line)
            line = clean_str(line)
            label = line[0]
            text = line[2:]
            text = text.lower()
            words = set(text.split())
            for word in words:
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            datum = {"y": int(label), "text": text, "num_words": len(text.split()), "split": 1}
            train_data.append(datum)

    with open(path + "stsa.fine.dev.txt", "r", encoding='utf-8-sig') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = str(line)
            line = clean_str(line)
            label = line[0]
            text = line[2:]
            text = text.lower()
            words = set(text.split())
            for word in words:
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            datum = {"y": int(label), "text": text, "num_words": len(text.split()), "split": 2}
            dev_data.append(datum)

    with open(path + "stsa.fine.test.txt", "r", encoding='utf-8-sig') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = str(line)
            line = clean_str(line)
            label = line[0]
            text = line[2:]
            text = text.lower()
            words = set(text.split())
            for word in words:
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            datum = {"y": int(label), "text": text, "num_words": len(text.split()), "split": 0}
            test_data.append(datum)

    return train_data, test_data, dev_data, vocab

def read_SST2_data(path):
    train_data = []
    test_data = []
    dev_data = []
    vocab = {}

    with open(path + "stsa.binary.train.txt", "r", encoding='utf-8-sig') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = str(line)
            line = clean_str(line)
            label = line[0]
            text = line[2:]
            text = text.lower()
            words = set(text.split())
            for word in words:
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            datum = {"y": int(label), "text": text, "num_words": len(text.split()), "split": 1}
            train_data.append(datum)

    with open(path + "stsa.binary.phrases.train.txt", "r", encoding='utf-8-sig') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = str(line)
            line = clean_str(line)
            label = line[0]
            text = line[2:]
            text = text.lower()
            words = set(text.split())
            for word in words:
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            datum = {"y": int(label), "text": text, "num_words": len(text.split()), "split": 1}
            train_data.append(datum)

    with open(path + "stsa.binary.dev.txt", "r", encoding='utf-8-sig') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = str(line)
            line = clean_str(line)
            label = line[0]
            text = line[2:]
            text = text.lower()
            words = set(text.split())
            for word in words:
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            datum = {"y": int(label), "text": text, "num_words": len(text.split()), "split": 2}
            dev_data.append(datum)

    with open(path + "stsa.binary.test.txt", "r", encoding='utf-8-sig') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = str(line)
            line = clean_str(line)
            label = line[0]
            text = line[2:]
            text = text.lower()
            words = set(text.split())
            for word in words:
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            datum = {"y": int(label), "text": text, "num_words": len(text.split()), "split": 0}
            test_data.append(datum)

    return train_data, test_data, dev_data, vocab

def load_bin_vec(fname, vocab):
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for _ in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                ch = "{:c}".format(ord(ch))
                if ch != '\n':
                    word.append(ch)
                if ch == ' ':
                    word = ''.join(word)
                    break
            word = word.strip()
            if word in vocab:
                word_vecs[word] = np.frombuffer(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25, k)

def get_W(word_vecs, k=300):
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype="f")
    i=1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1

    return W, word_idx_map

if __name__ == "__main__":
    # ./embedding/GoogleNews-vectors-negative300.bin
    w2v_file = sys.argv[1]
    corpus = sys.argv[2]
    print("loading data...")
    if corpus == "MR":
        data, vocab = read_MR_data("./data/MR/rt-polarity.pos", "./data/MR/rt-polarity.neg")
        max_l = np.max(pd.DataFrame(data)["num_words"])
        print("data loaded!")
        print("number of sentences: " + str(len(data)))
        print("vocab size: " + str(len(vocab)))
        print("max sentence length: " + str(max_l))
        print("loading word2vec vectors...", )
        w2v = load_bin_vec(w2v_file, vocab)
        print("word2vec loaded!")
        print("num words already in word2vec: " + str(len(w2v)))
        add_unknown_words(w2v, vocab)
        W, word_idx_map = get_W(w2v)
        rand_vecs = {}
        add_unknown_words(rand_vecs, vocab)
        W2, _ = get_W(rand_vecs)
        with open("mr.bin", "wb") as f:
            pickle.dump([data, W, W2, word_idx_map, vocab, max_l], f)
        print("dataset created!")

    if corpus == "CR":
        data, vocab = read_MR_data("./data/CR/custrev.pos.txt", "./data/CR/custrev.neg.txt")
        max_l = np.max(pd.DataFrame(data)["num_words"])
        print("data loaded!")
        print("number of sentences: " + str(len(data)))
        print("vocab size: " + str(len(vocab)))
        print("max sentence length: " + str(max_l))
        print("loading word2vec vectors...", )
        w2v = load_bin_vec(w2v_file, vocab)
        print("word2vec loaded!")
        print("num words already in word2vec: " + str(len(w2v)))
        add_unknown_words(w2v, vocab)
        W, word_idx_map = get_W(w2v)
        rand_vecs = {}
        add_unknown_words(rand_vecs, vocab)
        W2, _ = get_W(rand_vecs)
        with open("cr.bin", "wb") as f:
            pickle.dump([data, W, W2, word_idx_map, vocab, max_l], f)
        print("dataset created!")

    elif corpus == "TREC":
        train_data, test_data, vocab = read_TREC_data("./data/TREC/")
        max_l = np.max(pd.DataFrame(train_data)["num_words"])
        print("data loaded!")
        print("number of sentences: " + str(len(train_data)))
        print("vocab size: " + str(len(vocab)))
        print("max sentence length: " + str(max_l))
        print("loading word2vec vectors...", )
        w2v = load_bin_vec(w2v_file, vocab)
        print("word2vec loaded!")
        print("num words already in word2vec: " + str(len(w2v)))
        add_unknown_words(w2v, vocab)
        W, word_idx_map = get_W(w2v)
        rand_vecs = {}
        add_unknown_words(rand_vecs, vocab)
        W2, _ = get_W(rand_vecs)
        with open("trec.bin", "wb") as f:
            pickle.dump([train_data, test_data, W, W2, word_idx_map, vocab, max_l], f)
        print("dataset created!")

    elif corpus == "Subj":
        data, vocab = read_Subj_data("./data/Subj/quote.tok.gt9.5000", "./data/Subj/plot.tok.gt9.5000")
        max_l = np.max(pd.DataFrame(data)["num_words"])
        print("data loaded!")
        print("number of sentences: " + str(len(data)))
        print("vocab size: " + str(len(vocab)))
        print("max sentence length: " + str(max_l))
        print("loading word2vec vectors...", )
        w2v = load_bin_vec(w2v_file, vocab)
        print("word2vec loaded!")
        print("num words already in word2vec: " + str(len(w2v)))
        add_unknown_words(w2v, vocab)
        W, word_idx_map = get_W(w2v)
        rand_vecs = {}
        add_unknown_words(rand_vecs, vocab)
        W2, _ = get_W(rand_vecs)
        with open("subj.bin", "wb") as f:
            pickle.dump([data, W, W2, word_idx_map, vocab, max_l], f)
        print("dataset created!")

    elif corpus == "MPQA":
        data, vocab = read_MPQA_data("./data/MPQA/mpqa.pos.txt", "./data/MPQA/mpqa.neg.txt")
        max_l = np.max(pd.DataFrame(data)["num_words"])
        print("data loaded!")
        print("number of sentences: " + str(len(data)))
        print("vocab size: " + str(len(vocab)))
        print("max sentence length: " + str(max_l))
        print("loading word2vec vectors...", )
        w2v = load_bin_vec(w2v_file, vocab)
        print("word2vec loaded!")
        print("num words already in word2vec: " + str(len(w2v)))
        add_unknown_words(w2v, vocab)
        W, word_idx_map = get_W(w2v)
        rand_vecs = {}
        add_unknown_words(rand_vecs, vocab)
        W2, _ = get_W(rand_vecs)
        with open("mpqa.bin", "wb") as f:
            pickle.dump([data, W, W2, word_idx_map, vocab, max_l], f)
        print("dataset created!")

    elif corpus == "SST1":
        train_data, test_data, dev_data, vocab = read_SST1_data("./data/SST1/")
        max_l = np.max(pd.DataFrame(train_data)["num_words"])
        print("data loaded!")
        print("number of sentences: " + str(len(train_data)))
        print("vocab size: " + str(len(vocab)))
        print("max sentence length: " + str(max_l))
        print("loading word2vec vectors...", )
        w2v = load_bin_vec(w2v_file, vocab)
        print("word2vec loaded!")
        print("num words already in word2vec: " + str(len(w2v)))
        add_unknown_words(w2v, vocab)
        W, word_idx_map = get_W(w2v)
        rand_vecs = {}
        add_unknown_words(rand_vecs, vocab)
        W2, _ = get_W(rand_vecs)
        with open("sst1.bin", "wb") as f:
            pickle.dump([train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l], f)
        print("dataset created!")

    elif corpus == "SST2":
        train_data, test_data, dev_data, vocab = read_SST2_data("./data/SST2/")
        print(train_data[:4])
        print(test_data[:4])
        max_l = np.max(pd.DataFrame(train_data)["num_words"])
        print("data loaded!")
        print("number of sentences: " + str(len(train_data)))
        print("vocab size: " + str(len(vocab)))
        print("max sentence length: " + str(max_l))
        print("loading word2vec vectors...", )
        w2v = load_bin_vec(w2v_file, vocab)
        print("word2vec loaded!")
        print("num words already in word2vec: " + str(len(w2v)))
        add_unknown_words(w2v, vocab)
        W, word_idx_map = get_W(w2v)
        rand_vecs = {}
        add_unknown_words(rand_vecs, vocab)
        W2, _ = get_W(rand_vecs)
        with open("sst2.bin", "wb") as f:
            pickle.dump([train_data, test_data, dev_data, W, W2, word_idx_map, vocab, max_l], f)
        print("dataset created!")