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
import itertools
from scipy import stats

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

        x_val, y_val, sent_lengths_val, seq_lengths_val = data.fetch_test()
        feed_dict = {model.x: x_val, model.y: y_val, model.sent_lengths: sent_lengths_val,
                     model.seq_lengths: seq_lengths_val, model.dropout_keep_prob: 1,
                     model.max_seq_length: data.test_max_seq_length,
                     model.max_sent_length: data.test_max_sent_length
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
    selection = get_attention_weights(data, embds)
    visualize_attention(selection)
    tf.reset_default_graph()

    results = []
    now = "han_100d_163b_50cx_0.0001_0.5d"
    result = []
    for it in range(5):
        tf.reset_default_graph()
        with tf.Session() as sess:
            lstm = HierarchicalAttention(
                num_classes=2,
                vocab_size=embds.shape[0],
                embedding_size=embds.shape[1]
            )
        root_logdir = "logs"
        logdir = "{}/run-{}-{}/".format(root_logdir, now, it)
        acc, macro_f1, f1_0, f1_1, macro_precision, precision_0, precision_1, macro_recall, recall_0, recall_1 = evaluate(
            sess, data, embds, lstm, logdir)
        print(logdir)
        print(acc, " ", macro_f1, " ", f1_0, " ", f1_1, " ", macro_precision, " ", precision_0, " ",
              precision_1, " ", macro_recall, " ", recall_0, " ", recall_1)
        result.append(
            [acc, macro_f1, f1_0, f1_1, macro_precision, precision_0, precision_1, macro_recall,
             recall_0, recall_1])
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

    now = "han_100d_163b_50cx_0.0001_0.5d"
    with tf.Session() as sess:
        model = HierarchicalAttention(
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
    pred, a_word, a_sent = sess.run([predictions, model.alphas_word, model.alphas_sent], feed_dict=feed_dict)
    #pred, a1, A = sess.run([predictions, model.alphas1, model.alphas2, model.alphas3, model.alphas4],
                                    #feed_dict=feed_dict)
    a_word = np.reshape(a_word, [-1, data.val_max_seq_length, data.val_max_sent_length, 1])

    # filter on correct predictions
    zipped = list(zip(x_val, pred['labels'], pred['predictions'], pred['probabilities'], a_word, a_sent))
    # print(zipped[0:2])
    selection = [list(x) for x in zipped][133]
    zipped_correct = [list(x) for x in zipped if x[1]==x[2] and x[1] == 1]
    # print(zipped_correct[0:2])

    def get_predicted_prob(x):
        return (x[3])[(x[2])]

    sorted_correct = sorted(zipped_correct, key=get_predicted_prob, reverse=True)
    print(sorted_correct[0:2])

    #selection = sorted_correct[1]
    selection_zipped_tuple = list(zip(selection[0], selection[4], selection[5]))
    #selection_zipped_tuple = list(zip(selection[0], selection[4]))
    selection_zipped = [list(x) for x in selection_zipped_tuple]
    for s in selection_zipped:
        s[0] = dp.translate_to_voc(s[0])
    return selection_zipped

def visualize_attention(data):
    #data = np.array(data)
    data.reverse()
    attention_weights_word = np.array([np.squeeze(x[1]) for x in data])
    attention_weights_sent = np.array([np.squeeze(x[2]) for x in data])
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
        attention_weights_word[i, idx:] = 0
        # attention_weights3[i, idx:] = 0
        # attention_weights4[i, idx:] = 0
        if idx > max_idx:
            max_idx = idx
        if idx == 0:
            empty_rows += 1
    sentence = sentence[empty_rows:, 0:max_idx]
    attention_weights_word = attention_weights_word[empty_rows:, 0:max_idx]
    attention_weights_sent = attention_weights_sent[empty_rows:]
    # attention_weights3 = attention_weights3[empty_rows:, 0:max_idx]
    # attention_weights4 = attention_weights4[empty_rows:, 0:max_idx]

    max_weight1 = attention_weights_word.max()
    attention_weights_word = attention_weights_word / max_weight1  # increase weights to make visualization clearer
    max_weight2 = attention_weights_sent.max()
    attention_weights_sent = attention_weights_sent / max_weight2  # increase weights to make visualization clearer
    # max_weight3 = attention_weights3.max()
    # attention_weights3 = attention_weights3 / max_weight3  # increase weights to make visualization clearer
    # max_weight4 = attention_weights4.max()
    # attention_weights4 = attention_weights4 / max_weight4  # increase weights to make visualization clearer

    #print(np.shape(attention_weights1))
    print(np.shape(sentence))
    #print(np.shape(labels))

    MAX_FONTSIZE = 15
    MIN_FONTSIZE = 10

    def _font_size(word_len):
        return max(int(round(MAX_FONTSIZE - 0.5 * word_len)), MIN_FONTSIZE)

    def plot_attention(fname, attention_weights, attention_weights_sent, sentence):

        length = np.vectorize(lambda s: len(s))
        max_word_len = length(sentence).max()
        font_size_max_len = _font_size(max_word_len)

        plt.figure(figsize=(
        attention_weights.shape[-1] * (max_word_len * font_size_max_len / 100 + 0.5), attention_weights.shape[0]))

        plt.title("Attention")
        plt.xlabel("words")
        plt.ylabel("batch")

        pc_sent = plt.pcolor(attention_weights_sent, edgecolors='k', linewidths=4, cmap='Reds', vmin=0.0, vmax=1.0)
        pc_sent.update_scalarmappable()

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
        idx = [i + 0.5 for i in range(attention_weights_sent.shape[0])]
        plt.yticks(idx, attention_weights_sent)

        for l, i in zip(ax.yaxis.get_ticklabels(),  pc_sent.get_facecolors()):
            l.set_color(i)
            l.set_backgroundcolor(i)
            l.set_fontsize(15)

        plt.colorbar(pc)
        plt.savefig(fname)

    plot_attention("attention_real_han.png", attention_weights_word, np.array([[x] for x in attention_weights_sent]), sentence)
    # plot_attention("attention_real3.png", attention_weights3, sentence)
    # plot_attention("attention_real4.png", attention_weights4, sentence)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_confusion(data, embds):
    tf.reset_default_graph()

    now = "banl2norm_100d_163b_[10,10,10,10]cx_0.0001_0.5d_accstop"
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = BucketizedAttention(
            num_classes=2,
            vocab_size=embds.shape[0],
            embedding_size=embds.shape[1]
        )
    root_logdir = "logs"
    logdir = "{}/run-{}-{}/".format(root_logdir, now, 0)

    checkpoint_dir = "{}checkpoints".format(logdir)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: embds})
    saver.restore(sess, checkpoint_dir)
    predictions = model.predict()

    x_val, y_val, sent_lengths_val, seq_lengths_val = data.fetch_test()
    feed_dict = {model.x: x_val, model.y: y_val, model.sent_lengths: sent_lengths_val,
                 model.seq_lengths: seq_lengths_val, model.dropout_keep_prob: 1,
                 model.max_seq_length: data.test_max_seq_length,
                 model.max_sent_length: data.test_max_sent_length
                 }
    pred = sess.run(predictions, feed_dict=feed_dict)
    def fn(x):
        if x == 0:
            return 3
        elif x == 1:
            return 4
        elif x == 2:
            return 2
        elif x == 3:
            return 1
        elif x == 4:
            return 5
        elif x == 5:
            return 0
        else:
            return -1

    labels = list(map(fn, pred['labels']))
    predicts = list(map(fn, pred['predictions']))
    cnf_matrix = metrics.confusion_matrix(labels, predicts)
    # Plot non-normalized confusion matrix
    plt.figure()
    #classes = ["True", "Mostly-true", "Half-true", "Barely-true", "False", "Pants-on-fire"]
    classes = ["True", "False"]
    plot_confusion_matrix(cnf_matrix, classes=classes,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()


def t_test(data, embds):
    tf.reset_default_graph()

    acc_ban = []
    f1_ban = []
    now = "banl2norm_100d_163b_[10,10,10,10]cx_0.0001_0.5d_accstop"
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
        acc, macro_f1, f1_0, f1_1, macro_precision, precision_0, precision_1, macro_recall, recall_0, recall_1 = evaluate(
            sess, data, embds, lstm, logdir)
        print(acc)
        acc_ban.append(acc)
        f1_ban.append(macro_f1)

    tf.reset_default_graph()

    acc_cnn = [0.6313328137178488, 0.6157443491816056, 0.6110678098207326, 0.6141855027279813, 0.6165237724084178, 0.627435697583788, 0.6297739672642245, 0.6102883865939205, 0.6219797349961029, 0.6157443491816056, 0.6188620420888542, 0.6087295401402962, 0.6071706936866719, 0.6118472330475448, 0.6336710833982853, 0.6243180046765393, 0.6056118472330475, 0.6180826188620421, 0.6243180046765393, 0.6180826188620421, 0.6250974279033515, 0.6180826188620421, 0.6219797349961029, 0.6056118472330475, 0.6188620420888542, 0.6235385814497272, 0.6063912704598597, 0.5962587685113017, 0.6313328137178488, 0.6149649259547935]

    f1_cnn = [0.625208977558574, 0.6067531970160148, 0.6109316669026621, 0.6020553751990241, 0.6090837028412892, 0.6094950282209589, 0.6172590617767771, 0.607132008544496, 0.6080345191414308, 0.5998115849326153, 0.6085742361143607, 0.6078430656223209, 0.5935340795944845, 0.5862705332027911, 0.6173464207571212, 0.6042373835890662, 0.6010630976083375, 0.5991259035560702, 0.5946686067851712, 0.5925791031776069, 0.6052042516849045, 0.6115004325794092, 0.6152243182460431, 0.6045333820662768, 0.6009255107006212, 0.6008323601423038, 0.5949095710792511, 0.59088816113464, 0.6062203096074071, 0.6064241216914394]

    # now = "han_100d_163b_50cx_0.0001_0.5d"
    # for it in range(30):
    #     tf.reset_default_graph()
    #     with tf.Session() as sess:
    #         lstm = HierarchicalAttention(
    #             num_classes=2,
    #             vocab_size=embds.shape[0],
    #             embedding_size=embds.shape[1]
    #         )
    #     root_logdir = "logs"
    #     logdir = "{}/run-{}-{}/".format(root_logdir, now, it)
    #     acc, macro_f1, f1_0, f1_1, macro_precision, precision_0, precision_1, macro_recall, recall_0, recall_1 = evaluate(
    #         sess, data, embds, lstm, logdir)
    #     print(acc)
    #     acc_han.append(acc)
    #     f1_han.append(macro_f1)

    print(stats.ttest_ind(acc_ban, acc_cnn, equal_var=False))
    print(stats.ttest_ind(f1_ban, f1_cnn, equal_var=False))