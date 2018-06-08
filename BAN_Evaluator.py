import tensorflow as tf
import numpy as np
from datetime import datetime
import os
from CNN import CNN
from LSTM import LSTM
from BiLSTM import BiLSTM
from SLAN import Attention
from HAN2 import HierarchicalAttention
import sklearn.metrics as metrics
import DataProcessor as dp
import matplotlib.pyplot as plt
import numpy as np
from math import floor
from BAN import BucketizedAttention

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def evaluate(sess, data, embds, model, logdir):
        checkpoint_dir = "{}checkpoints".format(logdir)
        saver = tf.train.Saver()
        #saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_dir))
        # Training model
        #training_op, global_step = model.optimize()
        sess.run(tf.global_variables_initializer())
        sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: embds})
        saver.restore(sess, checkpoint_dir)
        predictions = model.predict()

        #print("Evaluation:")

        x_val, y_val, sent_lengths_val, seq_lengths_val = data.fetch_val()
        feed_dict = {model.x: x_val, model.y: y_val, model.sent_lengths: sent_lengths_val,
                     model.seq_lengths: seq_lengths_val, model.dropout_keep_prob: 1,
                     model.max_seq_length: data.val_max_seq_length,
                     model.max_sent_length: data.val_max_sent_length
                     }
        pred = sess.run(predictions, feed_dict=feed_dict)

        acc = metrics.accuracy_score(pred['labels'], pred['predictions'])
        macro_f1 = metrics.f1_score(pred['labels'], pred['predictions'], average="macro")
        f1_0 = metrics.f1_score(pred['labels'], pred['predictions'], pos_label=0)
        f1_1 = metrics.f1_score(pred['labels'], pred['predictions'], pos_label=1)
        macro_precision = metrics.precision_score(pred['labels'], pred['predictions'], average="macro")
        precision_0 = metrics.precision_score(pred['labels'], pred['predictions'], pos_label=0)
        precision_1 = metrics.precision_score(pred['labels'], pred['predictions'], pos_label=1)
        macro_recall = metrics.recall_score(pred['labels'], pred['predictions'], average="macro")
        recall_0 = metrics.recall_score(pred['labels'], pred['predictions'], pos_label=0)
        recall_1 = metrics.recall_score(pred['labels'], pred['predictions'], pos_label=1)


        return (acc, macro_f1, f1_0, f1_1, macro_precision, precision_0, precision_1, macro_recall, recall_0, recall_1)
        #return (acc, macro_f1, 1, 1, macro_precision, 1, 1, macro_recall, 1, 1)


def run_std(data, embds):
    #selection = get_attention_weights(data, embds)
    #visualize_attention(selection)
    tf.reset_default_graph()


    results = []

    #now = "han_100d_163b_50cx_0.0001_0.5d"
    now = "banl2norm_100d_163b_[10,10,10,10]cx_0.0001_0.5d_accstop"
    result = []
    for it in range(30):
        tf.reset_default_graph()
        with tf.Session() as sess:
            lstm = BucketizedAttention(
                num_classes=2,
                vocab_size=embds.shape[0],
                embedding_size=embds.shape[1]
            )
        root_logdir = "logs"
        logdir = "{}/run-{}-{}/".format(root_logdir, now, it)
        acc, macro_f1, f1_0, f1_1, macro_precision, precision_0, precision_1, macro_recall, recall_0, recall_1 = evaluate(sess, data, embds, lstm, logdir)
        print(logdir)
        print(acc, " ", macro_f1, " ", f1_0, " ", f1_1, " ", macro_precision, " ", precision_0, " ", precision_1, " ", macro_recall, " ", recall_0, " ", recall_1)
        result.append([acc, macro_f1, f1_0, f1_1, macro_precision, precision_0, precision_1, macro_recall, recall_0, recall_1])
    result_averages = np.mean(result, axis=0)
    print(result_averages)
    result_stds = np.std(result, axis=0)
    print(result_stds)
    result = list(zip(result_averages, result_stds))
    result.insert(0, now)
    results.append(result)
    print(result)

    print("averages-------")
    print(results)
    print("------------")



def get_attention_weights(data, embds):
    tf.reset_default_graph()

    it = 0

    #now = "han_100d_163b_50cx_0.0001_0.5d"
    now = "banl2norm_100d_163b_[10,10,10,10]cx_0.0001_0.5d_accstop"
    with tf.Session() as sess:
        model = BucketizedAttention(
            num_classes=2,
            vocab_size=embds.shape[0],
            embedding_size=embds.shape[1]
        )
    root_logdir = "logs"
    logdir = "{}/run-{}-{}/".format(root_logdir, now, it)

    checkpoint_dir = "{}checkpoints".format(logdir)
    saver = tf.train.Saver()
    # saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_dir))
    # Training model
    # training_op, global_step = model.optimize()
    sess.run(tf.global_variables_initializer())
    sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: embds})
    saver.restore(sess, checkpoint_dir)
    predictions = model.predict()

    # print("Evaluation:")
    x_val, y_val, sent_lengths_val, seq_lengths_val = data.fetch_val()
    feed_dict = {model.x: x_val, model.y: y_val, model.sent_lengths: sent_lengths_val,
                 model.seq_lengths: seq_lengths_val, model.dropout_keep_prob: 1,
                 model.max_seq_length: data.val_max_seq_length,
                 model.max_sent_length: data.val_max_sent_length
                 }
    pred, a1, a2, a3, a4 = sess.run([predictions, model.alphas1, model.alphas2, model.alphas3, model.alphas4], feed_dict=feed_dict)
    #pred, a1, A = sess.run([predictions, model.alphas1, model.alphas2, model.alphas3, model.alphas4],
                                    #feed_dict=feed_dict)
    a1 = np.reshape(a1, [-1, data.val_max_seq_length, data.val_max_sent_length, 1])
    a2 = np.reshape(a2, [-1, data.val_max_seq_length, data.val_max_sent_length, 1])
    a3 = np.reshape(a3, [-1, data.val_max_seq_length, data.val_max_sent_length, 1])
    a4 = np.reshape(a4, [-1, data.val_max_seq_length, data.val_max_sent_length, 1])
    # filter on correct predictions
    zipped = list(zip(x_val, pred['labels'], pred['predictions'], pred['probabilities'], a1, a2, a3, a4))
    selection = [list(x) for x in zipped][133]
    zipped_correct = [list(x) for x in zipped if x[1]==x[2] and x[1] == 1]


    def get_predicted_prob(x):
        return (x[3])[(x[2])]

    sorted_correct = sorted(zipped_correct, key=get_predicted_prob, reverse=True)
    print(sorted_correct[0:2])

    #selection = sorted_correct[1]
    selection_zipped_tuple = list(zip(selection[0], selection[4], selection[5], selection[6], selection[7]))
    #selection_zipped_tuple = list(zip(selection[0], selection[4]))
    selection_zipped = [list(x) for x in selection_zipped_tuple]
    for s in selection_zipped:
        s[0] = dp.translate_to_voc(s[0])
    return selection_zipped

def visualize_attention(data):
    #data = np.array(data)
    data.reverse()
    attention_weights1 = np.array([np.squeeze(x[1]) for x in data])
    attention_weights2 = np.array([np.squeeze(x[2]) for x in data])
    attention_weights3 = np.array([np.squeeze(x[3]) for x in data])
    attention_weights4 = np.array([np.squeeze(x[4]) for x in data])
    #max_weight = attention_weights.max()
    #attention_weights = attention_weights/max_weight # increase weights to make visualization clearer
    #max_weight1 = np.array(attention_weights1.max(axis=-1))
    #attention_weights1 = attention_weights1 / max_weight1[:, None]  # increase weights to make visualization clearer
    sentence = np.array([x[0] for x in data])
    #labels = np.array(["label-{}, pred-{}, prob-{}".format(x[1], x[2], max(x[3])) for x in data])
    max_idx = 0
    empty_rows = 0
    for i, s in enumerate(sentence):
        idx = list(s).index("PAD")
        attention_weights1[i, idx:] = 0
        attention_weights2[i, idx:] = 0
        attention_weights3[i, idx:] = 0
        attention_weights4[i, idx:] = 0
        if idx > max_idx:
            max_idx = idx
        if idx == 0:
            empty_rows += 1
    sentence = sentence[empty_rows:, 0:max_idx]
    attention_weights1 = attention_weights1[empty_rows:, 0:max_idx]
    attention_weights2 = attention_weights2[empty_rows:, 0:max_idx]
    attention_weights3 = attention_weights3[empty_rows:, 0:max_idx]
    attention_weights4 = attention_weights4[empty_rows:, 0:max_idx]

    max_weight1 = attention_weights1.max()
    attention_weights1 = attention_weights1 / max_weight1  # increase weights to make visualization clearer
    max_weight2 = attention_weights2.max()
    attention_weights2 = attention_weights2 / max_weight2  # increase weights to make visualization clearer
    max_weight3 = attention_weights3.max()
    attention_weights3 = attention_weights3 / max_weight3  # increase weights to make visualization clearer
    max_weight4 = attention_weights4.max()
    attention_weights4 = attention_weights4 / max_weight4  # increase weights to make visualization clearer

    print(np.shape(attention_weights1))
    print(np.shape(sentence))
    #print(np.shape(labels))

    MAX_FONTSIZE = 15
    MIN_FONTSIZE = 10

    def _font_size(word_len):
        return max(int(round(MAX_FONTSIZE - 0.5 * word_len)), MIN_FONTSIZE)

    def plot_attention(fname, attention_weights, sentence):

        length = np.vectorize(lambda s: len(s))
        max_word_len = length(sentence).max()
        font_size_max_len = _font_size(max_word_len)

        plt.figure(figsize=(
        attention_weights.shape[-1] * (max_word_len * font_size_max_len / 100 + 0.5), attention_weights.shape[0]))

        plt.title("Attention")
        plt.xlabel("words")
        plt.ylabel("batch")

        pc = plt.pcolor(attention_weights, edgecolors='k', linewidths=4, cmap='Blues', vmin=0.0, vmax=1.0)
        pc.update_scalarmappable()
        ax = pc.axes
        for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if np.all(color[:3] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            j, i = int(floor(x)), int(floor(y))
            if sentence[i, j] != "PAD":
                word = sentence[i, j]
            else:
                word = ""
            fontsize = _font_size(len(word))
            ax.text(x, y, word, ha="center", va="center", color=color, size=fontsize)
        #idx = [i + 0.5 for i in range(labels.shape[0])]
        #plt.yticks(idx, labels)

        plt.colorbar(pc)
        plt.savefig(fname)

    plot_attention("attention_real1.png", attention_weights1, sentence)
    plot_attention("attention_real2.png", attention_weights2, sentence)
    plot_attention("attention_real3.png", attention_weights3, sentence)
    plot_attention("attention_real4.png", attention_weights4, sentence)
