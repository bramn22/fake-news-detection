import tensorflow as tf
import numpy as np
from datetime import datetime
import os
from LSTM import LSTM
from BiLSTM import BiLSTM
from SLAN import Attention
from HAN import HierarchicalAttention
from CNN import CNN



config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def train(sess, data, embds, model, logdir, dropout_keep_prob):

    # Training model
    training_op, global_step = model.optimize()

    # Summaries
    loss_summary = tf.summary.scalar("loss", model.loss)
    acc_summary = tf.summary.scalar("accuracy", model.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = "{}/summaries/train".format(logdir)
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = "{}/summaries/dev".format(logdir)
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpointing
    checkpoint_dir = "{}/best.ckpt".format(logdir)
    saver = tf.train.Saver(tf.global_variables())

    sess.run(tf.global_variables_initializer())
    sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: embds})

    e = 0
    last_max = 0
    max_acc = 0
    while last_max <= 20 and e <= 200:
        sess.run(tf.local_variables_initializer())

        for b in range(data.n_batches):
            x_batch, y_batch, seq_lengths_train = data.fetch_batch(e, b)
            feed_dict = {model.x: x_batch, model.y: y_batch, model.seq_lengths: seq_lengths_train,
                         model.dropout_keep_prob: dropout_keep_prob}
            _, summaries = sess.run([training_op, train_summary_op], feed_dict=feed_dict)
            current_step = tf.train.global_step(sess, global_step)
            train_summary_writer.add_summary(summaries, current_step)
        print(e)
        print("Evaluation:")
        x_val, y_val, seq_lengths_val = data.fetch_val()
        feed_dict = {model.x: x_val, model.y: y_val, model.seq_lengths: seq_lengths_val,
                     model.dropout_keep_prob: 1}
        summaries, acc = sess.run(
            [dev_summary_op, model.accuracy], feed_dict=feed_dict)
        dev_summary_writer.add_summary(summaries, current_step)
        if acc > max_acc:
            print("new max:", max_acc, "->", acc)
            max_acc = acc
            last_max = 0
            saver.save(sess, checkpoint_dir)
        else:
            last_max = last_max + 1
        e = e + 1
    train_summary_writer.close()
    dev_summary_writer.close()

def run(data, embds):
    context_size = 50
    l2_scale = 1
    learning_rate_list = [0.001]
    dropout_keep_prob_list = [0.5]
    for learning_rate in learning_rate_list:
        for dropout_keep_prob in dropout_keep_prob_list:
            for it in range(25):
                tf.reset_default_graph()
                with tf.Session() as sess:
                    lstm = Attention(
                        num_classes=2,
                        vocab_size=embds.shape[0],
                        embedding_size=embds.shape[1],
                        learning_rate=learning_rate,
                        context_size=context_size
                    )
                    now = "att_100d_163b_{}cx_{}_{}d_{}l2".format(context_size, learning_rate, dropout_keep_prob, l2_scale)
                    root_logdir = "logs"
                    logdir = "{}/run-{}-{}/".format(root_logdir, now, 5+it)
                    train(sess, data, embds, lstm, logdir, dropout_keep_prob)



# def run(data, embds):
#     learning_rate_list = [0.001]
#     cell_sizes_list = [[50, 50]]
#     dropout_keep_prob_list = [0.5]
#
#     for learning_rate in learning_rate_list:
#         for cell_sizes in cell_sizes_list:
#             for dropout_keep_prob in dropout_keep_prob_list:
#                 for it in range(5):
#                     tf.reset_default_graph()
#                     with tf.Session() as sess:
#                         lstm = BiLSTM(
#                             num_classes=6,
#                             vocab_size=embds.shape[0],
#                             embedding_size=embds.shape[1],
#                             cell_sizes=cell_sizes
#                         )
#                         now = "bilstm_100d_163b_50_0.03"
#                         # now = "lstm_100d_163b_{}_{}_{}d".format(cell_sizes, learning_rate, dropout_keep_prob)
#                         root_logdir = "logs"
#                         logdir = "{}/run-{}-{}/".format(root_logdir, now, it)
#                         train(sess, data, embds, lstm, logdir, dropout_keep_prob)

# learning_rate_list = [0.001]
#     cell_sizes_list = [[50,50]]
#     dropout_keep_prob_list = [0.5]
#     for learning_rate in learning_rate_list:
#         for cell_sizes in cell_sizes_list:
#             for dropout_keep_prob in dropout_keep_prob_list:
#                 for it in range(2):
#                     tf.reset_default_graph()
#                     with tf.Session() as sess:
#                         lstm = CNN(
#                             seq_length=30,
#                             num_classes=2,
#                             vocab_size=embds.shape[0],
#                             embedding_size=embds.shape[1],
#                             filter_sizes=[3, 4, 5],
#                             num_filters= 100
#                         )
#                         now = "cnn_100d_163b_[3,4,5]_100_0.001_0.5d"
#                         root_logdir = "logs"
#                         logdir = "{}/run-{}-{}/".format(root_logdir, now, it)
#                         train(sess, data, embds, lstm, logdir, dropout_keep_prob)


# learning_rate_list = [0.001]
# cell_sizes_list = [[50, 50]]
# dropout_keep_prob_list = [0.5]
# for learning_rate in learning_rate_list:
#     for cell_sizes in cell_sizes_list:
#         for dropout_keep_prob in dropout_keep_prob_list:
#             for it in range(5):
#                 tf.reset_default_graph()
#                 with tf.Session() as sess:
#                     lstm = LSTM(
#                         num_classes=6,
#                         vocab_size=embds.shape[0],
#                         embedding_size=embds.shape[1],
#                         learning_rate=learning_rate,
#                         cell_sizes=cell_sizes
#                     )
#                     now = "lstm_100d_163b_{}_{}_{}d_6c".format(cell_sizes, learning_rate, dropout_keep_prob)
#                     root_logdir = "logs"
#                     logdir = "{}/run-{}-{}/".format(root_logdir, now, it)
#                     train(sess, data, embds, lstm, logdir, dropout_keep_prob)