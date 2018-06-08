import numpy as np
import sklearn
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from nltk import WordPunctTokenizer
from nltk import TreebankWordTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.moses import MosesTokenizer
from nltk.tokenize import sent_tokenize
import re

vocab = []
embds = []
embds_dim = 100

def get_wordvec_dict():
    load_wordvec_dict()
    return vocab, np.asarray(embds)

def translate_to_voc(array):
    text = []
    for i in array:
        text.append(vocab[i])
    return text

def load_wordvec_dict():
    # load the whole embedding into memory
    if not embds:
        f = open('../wordvecs/glove.6B.100d.txt', encoding="utf8")
        vocab.append("PAD")
        embds.append([0]*embds_dim)
        for line in f:
            values = line.split()
            vocab.append(values[0])
            embds.append(values[1:])
        f.close()
        print('Loaded %s word vectors.' % len(embds))

def clean_string(string):
    string = re.sub('[0-9]*(\.|\,)?[0-9]+', '167 ', string)
    string = re.sub('-', ' - ', string)
    string = re.sub("\'", " \' ", string)
    return string

def pad_sequence(length, padding_val, sequence):
    if length > len(sequence):
        sequence = np.concatenate((sequence, ([padding_val] * (length - np.shape(sequence)[0]))))
        #sequence.extend([padding_val] * (max_length - len(sequence)))
    elif length < len(sequence):
        del sequence[(length-len(sequence)):]
    return np.asarray(sequence)


def pad_sequences(sequences, max_length_allowed, length=-1, padding_val=0):
    seq_lengths = list(map(len, sequences))
    if max_length_allowed != -1:
        seq_lengths = [min(s, max_length_allowed) for s in seq_lengths]
    max_length = max(seq_lengths)
    if length == -1:
        if max_length_allowed != -1 and max_length_allowed < max_length:
            max_length = max_length_allowed
    else:
        max_length = length
    sequences = [pad_sequence(length=max_length, padding_val=padding_val, sequence=seq) for seq in sequences]
    # Sequence lengths contain original sequence lengths, not padded nor cropped!
    return np.asarray(sequences), np.asarray(seq_lengths)


def process_data(sequences_text):
    load_wordvec_dict()
    t = MosesTokenizer()
    sequences = np.empty_like(sequences_text)
    num_unrecognized = 0
    unrecognized_words = {}
    for i, s in enumerate(sequences_text):
        s = clean_string(s)
        s_t = t.tokenize(s, escape=False)
        s_t = [w.lower() for w in s_t]
        for j, w in enumerate(s_t):
            try:
                s_t[j] = vocab.index(w)
            except ValueError:
                # add vocabulary item
                vocab.append(w)
                # add embeddings item
                embds.append([0] * embds_dim)
                s_t[j] = len(vocab) - 1
                num_unrecognized += 1
                unrecognized_words[w] = 1
        sequences[i] = s_t
    print("Unrecognized vectors:::", num_unrecognized)
    print("Unrecognized words:::", unrecognized_words.keys())
    print("Processing Data Finished")
    return sequences

def process_hierarchical_data(sequences):
    load_wordvec_dict()

    t = MosesTokenizer()
    processed_sequences = np.zeros_like(sequences)
    for i, seq in enumerate(sequences):
        seq = clean_string(seq)
        sentences = sent_tokenize(seq)
        for z, sent in enumerate(sentences):
            sent_t = t.tokenize(sent)
            sent_t = [w.lower() for w in sent_t]
            for j, w in enumerate(sent_t):
                try:
                    sent_t[j] = vocab.index(w)
                except ValueError:
                    # add vocabulary item
                    vocab.append(w)
                    # add embeddings item
                    embds.append([0] * embds_dim)
                    sent_t[j] = len(vocab) - 1

            sentences[z] = sent_t
        processed_sequences[i] = sentences
    seq_lengths = np.asarray(list(map(len, processed_sequences)))
    sent_lengths = np.asarray([list(map(len, seq)) for seq in processed_sequences])
    sent_lengths = pad_sequences(sent_lengths, max_length_allowed=100)[0]
    print("seq_length shape: ")
    print(seq_lengths.shape)
    print(seq_lengths[0:3])
    print("sent_length shape: ")
    print(sent_lengths.shape)
    print(sent_lengths[0:3])
    print("max_sent_length")
    print(sent_lengths.max())
    max_seq_length = seq_lengths.max()
    max_sent_length = sent_lengths.max() # weird that max returns a list

    processed_sequences = np.asarray([pad_sequences(seq, max_length_allowed=max_sent_length, length=max_sent_length, padding_val=0)[0] for seq in processed_sequences])
    processed_sequences = pad_sequences(processed_sequences, max_length_allowed=max_seq_length, length=max_seq_length, padding_val=np.zeros_like(processed_sequences[0])[0])[0]

    print("Processing Data Finished")
    return processed_sequences, sent_lengths, seq_lengths


class Data:
    def __init__(self, train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y, batch_size=-1, shuffle=True, max_seq_length=-1, fixed_padding=False):
        self.prev_epoch = 0
        self.shuffle = shuffle

        self.train_set_x, self.seq_lengths_train = pad_sequences(train_set_x, max_seq_length)
        self.train_set_y = train_set_y
        length = -1
        if fixed_padding:
            length = self.train_set_x.shape[1]
        self.val_set_x, self.seq_lengths_val = pad_sequences(val_set_x, max_seq_length, length=length)
        self.val_set_y = val_set_y
        self.test_set_x, self.seq_lengths_test = pad_sequences(test_set_x, max_seq_length, length=length)
        self.test_set_y = test_set_y

        self.n_batches = 1
        if batch_size != -1:
            self.n_batches = int(np.ceil(self.train_set_x.shape[0] / batch_size))

        self.x_batches, self.y_batches, self.seq_lengths_batches = self.create_batches(self.train_set_x, self.train_set_y, self.seq_lengths_train)

    def fetch_batch(self, epoch, batch_idx):
        if epoch != self.prev_epoch and self.shuffle:
            x_shuffled, y_shuffled, seq_lengths_shuffled = sklearn.utils.shuffle(self.train_set_x, self.train_set_y,
                                                                                 self.seq_lengths_train)
            self.x_batches, self.y_batches, self.seq_lengths_batches = self.create_batches(x_shuffled, y_shuffled,
                                                                                           seq_lengths_shuffled)
            self.prev_epoch = epoch
        return self.x_batches[batch_idx], self.y_batches[batch_idx], self.seq_lengths_batches[batch_idx]

    def create_batches(self, x, y, seq_lengths):
        return np.split(x, self.n_batches), np.split(y, self.n_batches), np.split(seq_lengths, self.n_batches)

    def fetch_val(self):
        return self.val_set_x, self.val_set_y, self.seq_lengths_val

    def fetch_test(self):
        return self.test_set_x, self.test_set_y, self.seq_lengths_test

class Data2:
    def __init__(self, train_set, val_set, test_set, batch_size=-1, shuffle=True, max_seq_length=-1, fixed_padding=False):
        self.prev_epoch = 0
        self.shuffle = shuffle

        self.train_set_x, self.train_set_y, self.sent_lengths_train, self.seq_lengths_train = train_set
        self.val_set_x, self.val_set_y, self.sent_lengths_val, self.seq_lengths_val = val_set
        self.test_set_x, self.test_set_y, self.sent_lengths_test, self.seq_lengths_test = test_set

        self.train_max_seq_length = self.train_set_x.shape[1]
        self.train_max_sent_length= self.train_set_x.shape[2]
        self.val_max_seq_length = self.val_set_x.shape[1]
        self.val_max_sent_length = self.val_set_x.shape[2]
        self.test_max_seq_length = self.test_set_x.shape[1]
        self.test_max_sent_length = self.test_set_x.shape[2]
        self.n_batches = 1
        if batch_size != -1:
            self.n_batches = int(np.ceil(self.train_set_x.shape[0] / batch_size))
        print("train_x: {}, train_y: {}, sent_length: {}, seq_length: {}".format(self.train_set_x.shape, self.train_set_y.shape, self.sent_lengths_train.shape, self.seq_lengths_train.shape))
        self.x_batches, self.y_batches, self.sent_lengths_batches, self.seq_lengths_batches = self.create_batches(self.train_set_x, self.train_set_y, self.sent_lengths_train, self.seq_lengths_train)

    def fetch_batch(self, epoch, batch_idx):
        if epoch != self.prev_epoch and self.shuffle:
            print("train_x: {}, train_y: {}, sent_length: {}, seq_length: {}".format(self.train_set_x.shape,
                                                                                     self.train_set_y.shape,
                                                                                     self.sent_lengths_train.shape,
                                                                                     self.seq_lengths_train.shape))

            x_shuffled, y_shuffled, sent_lengths_shuffled, seq_lengths_shuffled = sklearn.utils.shuffle(self.train_set_x, self.train_set_y, self.sent_lengths_train,
                                                                                 self.seq_lengths_train)
            self.x_batches, self.y_batches, self.sent_lengths_batches, self.seq_lengths_batches = self.create_batches(x_shuffled, y_shuffled, sent_lengths_shuffled,
                                                                                           seq_lengths_shuffled)
            self.prev_epoch = epoch
        return self.x_batches[batch_idx], self.y_batches[batch_idx], self.sent_lengths_batches[batch_idx], self.seq_lengths_batches[batch_idx]

    def create_batches(self, x, y, sent_lengths, seq_lengths):
        return np.split(x, self.n_batches), np.split(y, self.n_batches), np.split(sent_lengths, self.n_batches), np.split(seq_lengths, self.n_batches)

    def fetch_val(self):
        return self.val_set_x, self.val_set_y, self.sent_lengths_val, self.seq_lengths_val

    def fetch_test(self):
        return self.test_set_x, self.test_set_y, self.sent_lengths_test, self.seq_lengths_test