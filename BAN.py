import tensorflow as tf
import numpy as np

class BucketizedAttention:
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
        sent_lengths_reshaped = tf.reshape(self.sent_lengths, [-1]) #TODO : check if correct!!!
        # vecs.shape = [batch_size*max_seq_length, max_sent_length, vec_dim]
        # word_attention.shape = [batch_size*max_seq_lengt, sent_vec_dim]
        with tf.variable_scope("wa1"):
            words_attention1 = self.word_attention1(x_reshaped, sent_lengths_reshaped)
        with tf.variable_scope("wa2"):
            words_attention2 = self.word_attention2(x_reshaped, sent_lengths_reshaped)
        with tf.variable_scope("wa3"):
            words_attention3 = self.word_attention3(x_reshaped, sent_lengths_reshaped)
        with tf.variable_scope("wa4"):
            words_attention4 = self.word_attention4(x_reshaped, sent_lengths_reshaped)

        # self.sentence_level.shape = [batch_size, max_seq_length, sent_vec_dim]
        self.sentence_level_unsummed1 = tf.reshape(words_attention1, [-1, self.max_seq_length, 20])
        self.sentence_level1 = tf.reduce_sum(self.sentence_level_unsummed1, axis=1)
        self.softmaxed1 = tf.nn.l2_normalize(self.sentence_level1, dim=-1)

        self.sentence_level_unsummed2 = tf.reshape(words_attention2, [-1, self.max_seq_length, 20])
        self.sentence_level2 = tf.reduce_sum(self.sentence_level_unsummed2, axis=1)
        self.softmaxed2 = tf.nn.l2_normalize(self.sentence_level2, dim=-1)

        self.sentence_level_unsummed3 = tf.reshape(words_attention3, [-1, self.max_seq_length, 20])
        self.sentence_level3 = tf.reduce_sum(self.sentence_level_unsummed3, axis=1)
        self.softmaxed3 = tf.nn.l2_normalize(self.sentence_level3, dim=-1)

        self.sentence_level_unsummed4 = tf.reshape(words_attention4, [-1, self.max_seq_length, 20])
        self.sentence_level4 = tf.reduce_sum(self.sentence_level_unsummed4, axis=1)
        self.softmaxed4 = tf.nn.l2_normalize(self.sentence_level4, dim=-1)

        #self.outputs = tf.concat([self.softmaxed1, self.softmaxed2, self.softmaxed3, self.softmaxed4], axis=-1)
        self.outputs = tf.concat([self.sentence_level1, self.sentence_level2, self.sentence_level3, self.sentence_level4], axis=-1)
        normalized_output = tf.nn.l2_normalize(self.outputs, dim=-1)
        shape = tf.shape(self.outputs)
        #tf.logging.info(msg=shape)
        #tf.logging.log(msg=shape)
        regularizer = tf.contrib.layers.l2_regularizer(scale=1.)
        with tf.name_scope("dense1"):
            self.dense1 = tf.layers.dense(
                inputs=normalized_output,
                units=50,
                activation=tf.nn.relu,
                kernel_regularizer=regularizer,

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
                inputs=normalized_output,
                units=num_classes,
                kernel_regularizer=regularizer
            )  # TODO: add initialization

        with tf.name_scope("loss"):
            xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            self.loss = tf.reduce_mean(xentropy)  # TODO: add l2 loss

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

    def word_attention1(self, sample, sent_length):
        self.word_context_vec = tf.Variable(tf.random_uniform(shape=[10, 1]), name="word_context_vec1")
        self.weight = tf.Variable(tf.random_uniform(shape=[20, 10]), name="sentence_weight1")
        self.bias = tf.Variable(tf.random_uniform(shape=[10]), name="sentence_bias1")
        vecs = tf.nn.embedding_lookup(self.W, sample)
        with tf.name_scope("word_biLSTM1"):
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(10, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(10, state_is_tuple=True)
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell,
                                                                             inputs=vecs,  sequence_length=sent_length, # TODO
                                                                             dtype=tf.float32, swap_memory=True, time_major=False)
            outputs = tf.concat([output_fw, output_bw], 2)

        with tf.name_scope("word_attention_layer1"):


            def fn(x):
                u = tf.tanh(tf.add(tf.matmul(x, self.weight), self.bias))  # u: (d, 50)
                dot = tf.matmul(u, self.word_context_vec)  # dot: (d, 1)
                return dot

            outputs_time_major = tf.transpose(outputs, perm=[1, 0, 2])
            dots_time_major = tf.map_fn(fn, outputs_time_major)
            dots = tf.transpose(dots_time_major, perm=[1, 0, 2])

            self.alphas1 = tf.nn.softmax(dots, dim=1)  # alphas: (d, t, 1)
            outputs_scaled = tf.multiply(self.alphas1, outputs)
            attention_outputs = tf.reduce_sum(outputs_scaled, axis=1)  # v: (d, e)
        return attention_outputs

    def word_attention2(self, sample, sent_length):
        self.word_context_vec2 = tf.Variable(tf.random_uniform(shape=[10, 1]), name="word_context_vec2")
        self.weight2 = tf.Variable(tf.random_uniform(shape=[20, 10]), name="sentence_weight2")
        self.bias2 = tf.Variable(tf.random_uniform(shape=[10]), name="sentence_bias2")
        vecs = tf.nn.embedding_lookup(self.W, sample)
        with tf.name_scope("word_biLSTM2"):
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(10, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(10, state_is_tuple=True)
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell,
                                                                             inputs=vecs, sequence_length=sent_length,
                                                                             # TODO
                                                                             dtype=tf.float32, swap_memory=True,
                                                                             time_major=False)
            outputs = tf.concat([output_fw, output_bw], 2)

        with tf.name_scope("word_attention_layer2"):


            def fn(x):
                u = tf.tanh(tf.add(tf.matmul(x, self.weight2), self.bias2))  # u: (d, 50)
                dot = tf.matmul(u, self.word_context_vec2)  # dot: (d, 1)
                return dot

            outputs_time_major = tf.transpose(outputs, perm=[1, 0, 2])
            dots_time_major = tf.map_fn(fn, outputs_time_major)
            dots = tf.transpose(dots_time_major, perm=[1, 0, 2])

            self.alphas2 = tf.nn.softmax(dots, dim=1)  # alphas: (d, t, 1)
            outputs_scaled = tf.multiply(self.alphas2, outputs)
            attention_outputs = tf.reduce_sum(outputs_scaled, axis=1)  # v: (d, e)
        return attention_outputs


    def word_attention3(self, sample, sent_length):
        self.word_context_vec3 = tf.Variable(tf.random_uniform(shape=[10, 1]), name="word_context_vec3")
        self.weight3 = tf.Variable(tf.random_uniform(shape=[20, 10]), name="sentence_weight3")
        self.bias3 = tf.Variable(tf.random_uniform(shape=[10]), name="sentence_bias3")
        vecs = tf.nn.embedding_lookup(self.W, sample)
        with tf.name_scope("word_biLSTM3"):
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(10, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(10, state_is_tuple=True)
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell,
                                                                             inputs=vecs, sequence_length=sent_length,
                                                                             # TODO
                                                                             dtype=tf.float32, swap_memory=True,
                                                                             time_major=False)
            outputs = tf.concat([output_fw, output_bw], 2)

        with tf.name_scope("word_attention_layer3"):


            def fn(x):
                u = tf.tanh(tf.add(tf.matmul(x, self.weight3), self.bias3))  # u: (d, 50)
                dot = tf.matmul(u, self.word_context_vec3)  # dot: (d, 1)
                return dot

            outputs_time_major = tf.transpose(outputs, perm=[1, 0, 2])
            dots_time_major = tf.map_fn(fn, outputs_time_major)
            dots = tf.transpose(dots_time_major, perm=[1, 0, 2])

            self.alphas3 = tf.nn.softmax(dots, dim=1)  # alphas: (d, t, 1)
            outputs_scaled = tf.multiply(self.alphas3, outputs)
            attention_outputs = tf.reduce_sum(outputs_scaled, axis=1)  # v: (d, e)
        return attention_outputs

    def word_attention4(self, sample, sent_length):
        self.word_context_vec4 = tf.Variable(tf.random_uniform(shape=[10, 1]), name="word_context_vec4")
        self.weight4 = tf.Variable(tf.random_uniform(shape=[20, 10]), name="sentence_weight4")
        self.bias4 = tf.Variable(tf.random_uniform(shape=[10]), name="sentence_bias4")
        vecs = tf.nn.embedding_lookup(self.W, sample)
        with tf.name_scope("word_biLSTM4"):
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(10, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(10, state_is_tuple=True)
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell,
                                                                             inputs=vecs, sequence_length=sent_length,
                                                                             # TODO
                                                                             dtype=tf.float32, swap_memory=True,
                                                                             time_major=False)
            outputs = tf.concat([output_fw, output_bw], 2)

        with tf.name_scope("word_attention_layer4"):


            def fn(x):
                u = tf.tanh(tf.add(tf.matmul(x, self.weight4), self.bias4))  # u: (d, 50)
                dot = tf.matmul(u, self.word_context_vec4)  # dot: (d, 1)
                return dot

            outputs_time_major = tf.transpose(outputs, perm=[1, 0, 2])
            dots_time_major = tf.map_fn(fn, outputs_time_major)
            dots = tf.transpose(dots_time_major, perm=[1, 0, 2])

            self.alphas4 = tf.nn.softmax(dots, dim=1)  # alphas: (d, t, 1)
            outputs_scaled = tf.multiply(self.alphas4, outputs)
            attention_outputs = tf.reduce_sum(outputs_scaled, axis=1)  # v: (d, e)
        return attention_outputs

