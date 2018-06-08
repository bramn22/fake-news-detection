import tensorflow as tf
import numpy as np

'''This HAN version was used for the final results.'''
class HierarchicalAttention:
    def __init__(self, num_classes, vocab_size, embedding_size):
        # self.x.shape = [batch_size, max_seq_length, max_sent_length]
        self.x = tf.placeholder(tf.int32, [None, None, None],
                                name="x")  # None because of difference in training and eval data dimensions
        self.y = tf.placeholder(tf.float32, [None, num_classes], name="y")  # TODO: why y: float and x: int??
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # self.sent_lengths.shape = [batch_size, max_seq_length]
        self.sent_lengths = tf.placeholder(tf.int32, [None, None])
        # self.seq_lengths.shape = [batch_size]
        self.seq_lengths = tf.placeholder(tf.int32, [None])
        self.max_sent_length = tf.placeholder(tf.int32)
        self.max_seq_length = tf.placeholder(tf.int32)

        self.W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), trainable=False, name="W")
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
        self.embedding_init = self.W.assign(self.embedding_placeholder)

        # x_reshaped.shape = [batch_sze*max_seq_length, max_sent_length]
        x_reshaped = tf.reshape(self.x, [-1, self.max_sent_length])
        sent_lengths_reshaped = tf.reshape(self.sent_lengths, [-1])
        # vecs.shape = [batch_size*max_seq_length, max_sent_length, vec_dim]
        # word_attention.shape = [batch_size*max_seq_lengt, sent_vec_dim]
        words_attention = self.word_attention(x_reshaped, sent_lengths_reshaped)

        # self.sentence_level.shape = [batch_size, max_seq_length, sent_vec_dim]
        self.sentence_level = tf.reshape(words_attention, [-1, self.max_seq_length, 100])
        with tf.variable_scope('rnn'):
            with tf.name_scope("sentence_biLSTM"):
                lstm_fw_cell = tf.contrib.rnn.LSTMCell(50, state_is_tuple=True)
                lstm_bw_cell = tf.contrib.rnn.LSTMCell(50, state_is_tuple=True)
                (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell,
                                                                                 inputs=self.sentence_level,
                                                                                 sequence_length=self.seq_lengths,
                                                                                 dtype=tf.float32, swap_memory=True,
                                                                                 time_major=False, scope='rnn')
                outputs = tf.concat([output_fw, output_bw], 2)

        with tf.name_scope("sentence_attention_layer"):
            weight = tf.Variable(tf.random_uniform(shape=[100, 50]))
            bias = tf.Variable(tf.random_uniform(shape=[50]))
            sentence_context_vec = tf.Variable(tf.random_uniform(shape=[50, 1]), name="sentence_context_vec")

            def fn(x):
                u = tf.tanh(tf.add(tf.matmul(x, weight), bias))  # u: (d, 50)
                dot = tf.matmul(u, sentence_context_vec)  # dot: (d, 1)
                return dot

            outputs_time_major = tf.transpose(outputs, perm=[1, 0, 2])
            dots_time_major = tf.map_fn(fn, outputs_time_major)
            dots = tf.transpose(dots_time_major, perm=[1, 0, 2])

            self.alphas_sent = tf.nn.softmax(dots, dim=1)  # alphas: (d, t, 1)
            outputs_scaled = tf.multiply(self.alphas_sent, outputs)
            attention_outputs = tf.reduce_sum(outputs_scaled, axis=1)  # v: (d, e)

        #regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
        with tf.name_scope("dense1"):
            self.dense1 = tf.layers.dense(
                inputs=attention_outputs,
                units=50,
                activation=tf.nn.relu

            )
        with tf.name_scope("dropout"):
            dropout = tf.layers.dropout(
                inputs=self.dense1,
                rate=self.dropout_keep_prob
            )
        # with tf.name_scope("dense2"):
        #     self.dense2 = tf.layers.dense(
        #         inputs=dropout,
        #         units=50,
        #         activation=tf.nn.relu,
        #         kernel_regularizer=regularizer
        #     )
        with tf.name_scope("output"):
            self.logits = tf.layers.dense(
                inputs=attention_outputs,
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
        learning_rate = 0.0001
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(self.loss, global_step=global_step), global_step

    def word_attention(self, sample, sent_length):
        self.word_context_vec = tf.Variable(tf.random_uniform(shape=[50, 1]), name="word_context_vec")
        self.weight = tf.Variable(tf.random_uniform(shape=[100, 50]), name="sentence_weight")
        self.bias = tf.Variable(tf.random_uniform(shape=[50]), name="sentence_bias")
        vecs = tf.nn.embedding_lookup(self.W, sample)
        with tf.name_scope("word_biLSTM"):
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(50, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(50, state_is_tuple=True)
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell,
                                                                             inputs=vecs,  sequence_length=sent_length,
                                                                             dtype=tf.float32, swap_memory=True, time_major=False)
            outputs = tf.concat([output_fw, output_bw], 2)

        with tf.name_scope("word_attention_layer"):


            def fn(x):
                u = tf.tanh(tf.add(tf.matmul(x, self.weight), self.bias))  # u: (d, 50)
                dot = tf.matmul(u, self.word_context_vec)  # dot: (d, 1)
                return dot

            outputs_time_major = tf.transpose(outputs, perm=[1, 0, 2])
            dots_time_major = tf.map_fn(fn, outputs_time_major)
            dots = tf.transpose(dots_time_major, perm=[1, 0, 2])

            self.alphas_word = tf.nn.softmax(dots, dim=1)  # alphas: (d, t, 1)
            outputs_scaled = tf.multiply(self.alphas_word, outputs)
            attention_outputs = tf.reduce_sum(outputs_scaled, axis=1)  # v: (d, e)
        return attention_outputs
