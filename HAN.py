import tensorflow as tf
import numpy as np

'''This is the first HAN version using a while loop. This model was not used since it was too computationally expensive.'''
class HierarchicalAttention:
    def __init__(self, num_classes, vocab_size, embedding_size, cell_sizes):

        self.x = tf.placeholder(tf.int32, [None, None, None],
                                name="x")  # None because of difference in training and eval data dimensions
        self.y = tf.placeholder(tf.float32, [None, num_classes], name="y")  # TODO: why y: float and x: int??
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.sent_lengths = tf.placeholder(tf.int32, [None, None])
        self.seq_lengths = tf.placeholder(tf.int32, [None])

        self.W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), trainable=False, name="W")
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
        self.embedding_init = self.W.assign(self.embedding_placeholder)


        self.x_rows = tf.shape(self.x)[0]
        self.x_pre = tf.TensorArray(tf.int32, size=self.x_rows)
        self.x_pre = self.x_pre.unstack(self.x)
        self.x_post = tf.TensorArray(tf.float32, size=self.x_rows)
        self.sent_lengths_ta = tf.TensorArray(tf.int32, size=self.x_rows)
        self.sent_lengths_ta = self.sent_lengths_ta.unstack(self.sent_lengths)
        i = tf.constant(1)
        self.word_context_vec = tf.Variable(tf.random_uniform(shape=[50, 1]), name="word_context_vec")
        self.weight = tf.Variable(tf.random_uniform(shape=[100, 50]), name="sentence_weight")
        self.bias = tf.Variable(tf.random_uniform(shape=[50]), name="sentence_bias")


        sentences = self.x_pre.read(0)
        sent_length = self.sent_lengths_ta.read(0)


        with tf.variable_scope("word_attenion2", reuse=None):
            sentences_vectorized = self.word_attention(sentences, sent_length)
        self.x_post = self.x_post.write(0, sentences_vectorized)

        def body(i, x):
            sentences = self.x_pre.read(i)
            sent_length = self.sent_lengths_ta.read(i)
            with tf.variable_scope("word_attenion2", reuse=True):
                sentences_vectorized = self.word_attention(sentences, sent_length)
            x = x.write(i, sentences_vectorized)
            return [tf.add(i, 1), x]

        def condition(i, x):
            return tf.less(i, self.x_rows)
        # read_row = ta.read(row_ix)
        # x_post = x_post.write(row_ix, vector)

        #with tf.device("/cpu:0"):
        loop = tf.while_loop(condition, body, [i, self.x_post], parallel_iterations=200, swap_memory=True)

        self.sentence_level = loop[1].stack()
        print(tf.shape(self.sentence_level))
        #self.sentence_embeddings = tf.map_fn(self.word_attention, self.x)

        with tf.name_scope("sentence_biLSTM"):
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(50, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(50, state_is_tuple=True)
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell,
                                                                             inputs=self.sentence_level,
                                                                             sequence_length=self.seq_lengths,
                                                                             dtype=tf.float32, swap_memory=True,
                                                                             time_major=False)
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

            alphas = tf.nn.softmax(dots, dim=1)  # alphas: (d, t, 1)
            outputs_scaled = tf.multiply(alphas, outputs)
            attention_outputs = tf.reduce_sum(outputs_scaled, axis=1)  # v: (d, e)

        with tf.name_scope("dense"):
            self.dense = tf.layers.dense(
                inputs=attention_outputs,
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
            "classes": tf.argmax(input=self.logits, axis=1),
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
        vecs = tf.nn.embedding_lookup(self.W, sample)
        with tf.name_scope("word_biLSTM"):
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(50, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(50, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
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

            alphas = tf.nn.softmax(dots, dim=1)  # alphas: (d, t, 1)
            outputs_scaled = tf.multiply(alphas, outputs)
            attention_outputs = tf.reduce_sum(outputs_scaled, axis=1)  # v: (d, e)
        return attention_outputs
