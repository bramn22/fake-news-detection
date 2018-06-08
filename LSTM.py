import tensorflow as tf
import numpy as np

class LSTM:
    def __init__(self, num_classes, vocab_size, embedding_size, learning_rate=0.001, cell_sizes=[50]):
        self.learning_rate = learning_rate
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

        with tf.name_scope("multiRNN"):
            #cell_1 = tf.contrib.rnn.BasicLSTMCell(num_units=50, state_is_tuple=True)
            #cell_2 = tf.contrib.rnn.BasicLSTMCell(num_units=50, state_is_tuple=True)
            #cells = [cell_1, cell_2]
            cells = [tf.contrib.rnn.BasicLSTMCell(num_units=num, state_is_tuple=True) for num in cell_sizes]
            cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob) for cell in cells]

            multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells)
            outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, self.vecs, dtype=tf.float32, sequence_length=self.seq_lengths, swap_memory=True)

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(
                inputs=states[1][1],
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
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(self.loss, global_step=global_step), global_step
