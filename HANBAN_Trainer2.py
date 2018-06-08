import tensorflow as tf
import numpy as np
from datetime import datetime
import os
from LSTM import LSTM
from BiLSTM import BiLSTM
from SLAN import Attention
from HAN2 import HierarchicalAttention
from BAN import BucketizedAttention
from CNN import CNN



config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def train(sess, data, embds, model, logdir):
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
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir)

    # Checkpointing
    checkpoint_dir = "{}/checkpoints".format(logdir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    sess.run(tf.global_variables_initializer())
    sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: embds})

    e = 0
    last_max = 0
    max_acc = 0
    while last_max <= 20 and e <= 200:
        sess.run(tf.local_variables_initializer())

        for b in range(data.n_batches):
            x_batch, y_batch, sent_lengths_train, seq_lengths_train = data.fetch_batch(e, b)
            feed_dict = {model.x: x_batch, model.y: y_batch, model.sent_lengths: sent_lengths_train,
                         model.seq_lengths: seq_lengths_train, model.dropout_keep_prob: 0.5,
                         model.max_seq_length: data.train_max_seq_length,
                         model.max_sent_length: data.train_max_sent_length}  # TODO: add dynamic sequence lengths
            _, summaries = sess.run([training_op, train_summary_op], feed_dict=feed_dict)
            current_step = tf.train.global_step(sess, global_step)
            train_summary_writer.add_summary(summaries, current_step)
        print(e)
        print("Evaluation:")
        x_val, y_val, sent_lengths_val, seq_lengths_val = data.fetch_val()
        feed_dict = {model.x: x_val, model.y: y_val, model.sent_lengths: sent_lengths_val,
                     model.seq_lengths: seq_lengths_val, model.dropout_keep_prob: 1,
                     model.max_seq_length: data.val_max_seq_length,
                     model.max_sent_length: data.val_max_sent_length
                     }
        summaries, accuracy, loss = sess.run([dev_summary_op, model.accuracy, model.loss], feed_dict=feed_dict)
        dev_summary_writer.add_summary(summaries, current_step)
        if accuracy > max_acc:
            print("new max:", max_acc, "->", accuracy)
            max_acc = accuracy
            last_max = 0
            saver.save(sess, checkpoint_dir)
        else:
            last_max = last_max + 1
        e = e + 1
    train_summary_writer.close()
    dev_summary_writer.close()

def run(data, embds):

    for it in range(25):
        tf.reset_default_graph()
        with tf.Session() as sess:
            lstm = BucketizedAttention(
                num_classes=2,
                vocab_size=embds.shape[0],
                embedding_size=embds.shape[1],
            )

            now = "run-banl2norm_100d_163b_[10,10,10,10]cx_0.0001_0.5d_accstop"
            root_logdir = "logs"
            logdir = "{}/run-{}-{}/".format(root_logdir, now, it+5)
            train(sess, data, embds, lstm, logdir)