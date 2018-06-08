import tensorflow as tf
import numpy as np
from datetime import datetime
import os
from CNN import CNN
from LSTM import LSTM
from BiLSTM import BiLSTM
from SLAN import Attention
from HAN import HierarchicalAttention
import sklearn.metrics as metrics
import DataProcessor as dp
import matplotlib.pyplot as plt
import numpy as np
from math import floor

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def evaluate(sess, data, embds, model, logdir):
        checkpoint_dir = "{}best.ckpt".format(logdir)
        saver = tf.train.Saver()
        #saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_dir))
        # Training model
        #training_op, global_step = model.optimize()
        sess.run(tf.global_variables_initializer())
        sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: embds})
        saver.restore(sess, checkpoint_dir)
        predictions = model.predict()

        #print("Evaluation:")
        x_val, y_val, seq_lengths_val = data.fetch_test()
        feed_dict = {model.x: x_val, model.y: y_val, model.seq_lengths: seq_lengths_val,
                     model.dropout_keep_prob: 1}
        loss, acc, pred = sess.run([model.loss, model.accuracy, predictions], feed_dict=feed_dict)

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

        # filter on correct predictions
        #zipped = list(zip(x_val, pred['labels'], pred['predictions'], pred['probabilities']))
        #print(zipped[0:2])
        #zipped_correct = [x for x in zipped if x[1]==x[2]]
        #print(zipped_correct[0:2])

        def get_predicted_prob(x):
            return (x[3])[(x[2])]
        return (acc, macro_f1, f1_0, f1_1, macro_precision, precision_0, precision_1, macro_recall, recall_0, recall_1)
        #return (acc, macro_f1, 1, 1, macro_precision, 1, 1, macro_recall, 1, 1)

        #sorted_correct = sorted(zipped_correct, key=get_predicted_prob, reverse=True)
        #print(sorted_correct[0:2])
        # order by highest probabilities

        #correct_predictions = np.equal(pred['labels'], pred['predictions'])

        # get highest certainty predictions



def run(data, embds):
    if False:
        run_std(data, embds)
    else:
        tf.reset_default_graph()

        results = []

        #now = "lstm_100d_163b_{}_{}_{}d_6c".format(cell_sizes, learning_rate, dropout_keep_prob)
        #now = "bilstm_100d_163b_50_0.03_6c"
        now = "cnn_100d_163b_[3,4,5]_100_0.001_0.5d"

        result = []
        for it in range(5):
            tf.reset_default_graph()
            with tf.Session() as sess:
                lstm = CNN(
                    seq_length=30,
                    num_classes=2,
                    vocab_size=embds.shape[0],
                    embedding_size=embds.shape[1],
                    filter_sizes=[3, 4, 5],
                    num_filters= 100
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


def run_std(data, embds):
    selection = get_attention_weights(data, embds)
    visualize_attention(np.array(selection))

    tf.reset_default_graph()

    learning_rate_list = [0.001]
    context_size_list = [50]
    dropout_keep_prob_list = [0.5]
    l2_scale_list = [1]

    results = []
    for learning_rate in learning_rate_list:
        for context_size in context_size_list:
            for dropout_keep_prob in dropout_keep_prob_list:
                for l2_scale in l2_scale_list:
                    now = "att_100d_163b_{}cx_{}_{}d_{}l2".format(context_size, learning_rate, dropout_keep_prob,
                                                                  l2_scale)
                    result = []
                    for it in range(5):
                        tf.reset_default_graph()
                        with tf.Session() as sess:
                            lstm = Attention(
                                num_classes=2,
                                vocab_size=embds.shape[0],
                                embedding_size=embds.shape[1],
                                learning_rate=learning_rate,
                                context_size=context_size
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

    learning_rate = 0.001
    context_size = 50
    dropout_keep_prob = 0.5
    l2_scale = 1
    it = 1

    now = "att_100d_163b_{}cx_{}_{}d_{}l2".format(context_size, learning_rate, dropout_keep_prob,
                                                                  l2_scale)
    with tf.Session() as sess:
        model = Attention(
            num_classes=2,
            vocab_size=embds.shape[0],
            embedding_size=embds.shape[1],
            learning_rate=learning_rate,
            context_size=context_size
        )
    root_logdir = "logs"
    logdir = "{}/run-{}-{}/".format(root_logdir, now, it)

    checkpoint_dir = "{}best.ckpt".format(logdir)
    saver = tf.train.Saver()
    # saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_dir))
    # Training model
    # training_op, global_step = model.optimize()
    sess.run(tf.global_variables_initializer())
    sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: embds})
    saver.restore(sess, checkpoint_dir)
    predictions = model.predict()

    # print("Evaluation:")
    x_val, y_val, seq_lengths_val = data.fetch_val()
    feed_dict = {model.x: x_val, model.y: y_val, model.seq_lengths: seq_lengths_val,
                 model.dropout_keep_prob: 1}
    loss, acc, pred, alphas = sess.run([model.loss, model.accuracy, predictions, model.alphas], feed_dict=feed_dict)

    # filter on correct predictions
    zipped = list(zip(x_val, pred['labels'], pred['predictions'], pred['probabilities'], alphas))
    # print(zipped[0:2])
    #zipped_correct = [list(x) for x in zipped if x[1]==x[2] and x[1]==0]
    zipped_correct = [[list(x) for x in zipped][133]]
    # print(zipped_correct[0:2])

    def get_predicted_prob(x):
        return (x[3])[(x[2])]

    sorted_correct = sorted(zipped_correct, key=get_predicted_prob, reverse=True)
    print(sorted_correct[0:2])

    selection = list(sorted_correct[0:20])
    for s in selection:
        s[0] = dp.translate_to_voc(s[0])
    return selection

def visualize_attention(data):
    data = np.array(data)
    attention_weights = np.array([np.squeeze(x[4]) for x in data])
    #max_weight = attention_weights.max()
    #attention_weights = attention_weights/max_weight # increase weights to make visualization clearer
    #max_weight = np.array(attention_weights.max(axis=-1))
    #attention_weights = attention_weights/max_weight[:, None] # increase weights to make visualization clearer
    sentence = np.array([x[0] for x in data])
    labels = np.array(["label-{}, pred-{}, prob-{}".format(x[1], x[2], max(x[3])) for x in data])

    max_idx = 0
    empty_rows = 0
    for i, s in enumerate(sentence):
        idx = list(s).index("PAD")
        attention_weights[i, idx:] = 0
        if idx > max_idx:
            max_idx = idx
        if idx == 0:
            empty_rows += 1
    sentence = sentence[empty_rows:, 0:max_idx]
    attention_weights = attention_weights[empty_rows:, 0:max_idx]

    max_weight = np.array(attention_weights.max(axis=-1))
    attention_weights = attention_weights/max_weight[:, None] # increase weights to make visualization clearer

    print(np.shape(attention_weights))
    print(np.shape(sentence))
    print(np.shape(labels))

    MAX_FONTSIZE = 15
    MIN_FONTSIZE = 10

    def _font_size(word_len):
        return max(int(round(MAX_FONTSIZE - 0.5 * word_len)), MIN_FONTSIZE)

    def plot_attention(fname, attention_weights, sentence, labels):

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
        idx = [i + 0.5 for i in range(labels.shape[0])]
        plt.yticks(idx, labels)

        plt.colorbar(pc)
        plt.savefig(fname)

    plot_attention("attention_real1.png", attention_weights, sentence, labels)

# learning_rate_list = [0.0001]
# cell_sizes_list = [[50, 50]]
# dropout_keep_prob_list = [0.2, 0.5, 0.8, 1]
# averages = []
# for learning_rate in learning_rate_list:
#     for cell_sizes in cell_sizes_list:
#         for dropout_keep_prob in dropout_keep_prob_list:
#             now = "lstm_100d_163b_{}_{}_{}d".format(cell_sizes, learning_rate, dropout_keep_prob)
#             average = [0., 0., 0., 0.]
#             for it in range(5):
#                 tf.reset_default_graph()
#                 with tf.Session() as sess:
#                     lstm = LSTM(
#                         num_classes=2,
#                         vocab_size=embds.shape[0],
#                         embedding_size=embds.shape[1],
#                         learning_rate=learning_rate,
#                         cell_sizes=cell_sizes
#                     )

# learning_rate_list = [0.0001]
#         cell_sizes_list = [[50, 50]]
#         dropout_keep_prob_list = [0.5]
#         results = []
#         for learning_rate in learning_rate_list:
#             for cell_sizes in cell_sizes_list:
#                 for dropout_keep_prob in dropout_keep_prob_list:
#                     now = "lstm_100d_163b_{}_{}_{}d".format(cell_sizes, learning_rate, dropout_keep_prob)
#                     #now = "bilstm_100d_163b_50_0.03"
#                     average = [0., 0., 0., 0.]
#                     result = []
#                     for it in range(2):
#                         tf.reset_default_graph()
#                         with tf.Session() as sess:
#                             lstm = LSTM(
#                                 num_classes=2,
#                                 vocab_size=embds.shape[0],
#                                 embedding_size=embds.shape[1],
#                                 learning_rate=learning_rate,
#                                 cell_sizes=cell_sizes
#                             )
#                         root_logdir = "logs"
#                         logdir = "{}/run-{}-{}/".format(root_logdir, now, it)
#                         acc, f1, p, r = evaluate(sess, data, embds, lstm, logdir)
#                         print(logdir)
#                         print(acc, " ", f1, " ", p, " ", r)
#                         result.append([acc, f1, p, r])
#                     result_averages = np.mean(result, axis=0)
#                     print(result_averages)
#                     result_stds = np.std(result, axis=0)
#                     print(result_stds)
#                     result = list(zip(result_averages, result_stds))
#                     result.insert(0, now)
#                     results.append(result)
#                     print(result)
#
#         print("averages-------")
#         print(results)
#         print("------------")
#
# learning_rate_list = [0.0001]
#         cell_sizes_list = [[50, 50]]
#         dropout_keep_prob_list = [0.5]
#         results = []
#         for learning_rate in learning_rate_list:
#             for cell_sizes in cell_sizes_list:
#                 for dropout_keep_prob in dropout_keep_prob_list:
#                     #now = "lstm_100d_163b_{}_{}_{}d".format(cell_sizes, learning_rate, dropout_keep_prob)
#                     now = "cnn_100d_163b_[3,4,5]_100_0.001_0.5d"
#                     average = [0., 0., 0., 0.]
#                     result = []
#                     for it in range(5):
#                         tf.reset_default_graph()
#                         with tf.Session() as sess:
#                             lstm = CNN(
#                                 seq_length=30,
#                                 num_classes=2,
#                                 vocab_size=embds.shape[0],
#                                 embedding_size=embds.shape[1],
#                                 filter_sizes=[3, 4, 5],
#                                 num_filters= 100
#                             )

# lstm = BiLSTM(
#                                 num_classes=6,
#                                 vocab_size=embds.shape[0],
#                                 embedding_size=embds.shape[1]#,
#                                 #learning_rate=learning_rate,
#                                 #cell_sizes=cell_sizes
#                             )

def t_test(data, embds):
    tf.reset_default_graph()
    learning_rate = 0.001
    context_size = 50
    dropout_keep_prob = 0.5
    l2_scale = 1

    acc_cnn = []
    f1_cnn = []
    #now = "cnn_100d_163b_[3,4,5]_100_0.001_0.5d"
    for it in range(30):
        tf.reset_default_graph()
        with tf.Session() as sess:

            now = "att_100d_163b_{}cx_{}_{}d_{}l2".format(context_size, learning_rate, dropout_keep_prob,
                                                          l2_scale)
            with tf.Session() as sess:
                lstm = Attention(
                    num_classes=2,
                    vocab_size=embds.shape[0],
                    embedding_size=embds.shape[1],
                    learning_rate=learning_rate,
                    context_size=context_size
                )

        root_logdir = "logs"
        logdir = "{}/run-{}-{}/".format(root_logdir, now, it)
        acc, macro_f1, f1_0, f1_1, macro_precision, precision_0, precision_1, macro_recall, recall_0, recall_1 = evaluate(
            sess, data, embds, lstm, logdir)
        print(acc)
        acc_cnn.append(acc)
        f1_cnn.append(macro_f1)

    print(acc_cnn)
    print(f1_cnn)
