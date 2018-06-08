import tensorflow as tf
import numpy as np

class BiLSTM:
    def __init__(self, num_classes, vocab_size, embedding_size):
        self.x = tf.placeholder(tf.int32, [None, None],
                                name="x")
        self.y = tf.placeholder(tf.float32, [None, num_classes], name="y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.seq_lengths = tf.placeholder(tf.int32, [None])

        with tf.name_scope("embedding"):
            W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), trainable=False, name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
            self.embedding_init = W.assign(self.embedding_placeholder)
            self.vecs = tf.nn.embedding_lookup(W, self.x)

        with tf.name_scope("biRNN"):
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(50, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(50, state_is_tuple=True)
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell,
                                                                             inputs=self.vecs, sequence_length=self.seq_lengths,
                                                                             dtype=tf.float32, swap_memory=True)
            outputs = tf.concat([output_fw[:, -1, :], output_bw[:, 0, :]], 1)
            #outputs = output_fw
            print("output shape:-----------")
            print(np.shape(outputs))
        with tf.name_scope("dense"):
            self.dense = tf.layers.dense(
                inputs=outputs,
                units=75
            )
        with tf.name_scope("output"):
            self.logits = tf.layers.dense(
                inputs=self.dense,
                units=num_classes
            )

        with tf.name_scope("loss"):
            xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            self.loss = tf.reduce_mean(xentropy)

        with tf.name_scope("accuracy"):
            correct = tf.equal(tf.argmax(input=self.logits, axis=1), tf.argmax(input=self.y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    def predict(self):
        predictions = {
            "labels": tf.argmax(input=self.y, axis=1),
            "predictions": tf.argmax(input=self.logits, axis=1),
            "probabilities": tf.nn.softmax(self.logits)
        }
        return predictions

    def optimize(self):
        # Training model
        learning_rate = 0.03
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(self.loss, global_step=global_step), global_step
