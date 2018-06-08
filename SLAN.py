import tensorflow as tf
import numpy as np

class Attention:
    def __init__(self, num_classes, vocab_size, embedding_size, learning_rate, context_size):
        self.learning_rate = learning_rate

        self.x = tf.placeholder(tf.int32, [None, None],
                                name="x")
        self.y = tf.placeholder(tf.float32, [None, num_classes], name="y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.l2_scale = tf.placeholder(tf.float32, name="l2_scale")
        self.seq_lengths = tf.placeholder(tf.int32, [None])

        with tf.name_scope("embedding"):
            W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), trainable=False, name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
            self.embedding_init = W.assign(self.embedding_placeholder)
            self.vecs = tf.nn.embedding_lookup(W, self.x)

        with tf.name_scope("biLSTM"):
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(context_size, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(context_size, state_is_tuple=True)
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell,
                                                                             inputs=self.vecs, sequence_length=self.seq_lengths,
                                                                             dtype=tf.float32, swap_memory=True, time_major=False)
            outputs = tf.concat([output_fw, output_bw], 2)

        with tf.name_scope("attention"):
            weight1 = tf.Variable(tf.random_uniform(shape=[context_size*2, context_size]))
            bias1 = tf.Variable(tf.random_uniform(shape=[context_size]))
            #weight2 = tf.Variable(tf.random_uniform(shape=[context_size * 2, context_size]))
            #bias2 = tf.Variable(tf.random_uniform(shape=[context_size]))
            self.context_vec = tf.Variable(tf.random_uniform(minval=-1, shape=[context_size, 1]), name="context_vec")
            #self.context_vec2 = tf.Variable(tf.random_uniform(minval=-1, shape=[context_size, 1]), name="context_vec2")

            def fn1(x):
                u = tf.tanh(tf.add(tf.matmul(x, weight1), bias1))  # u: (d, 50)
                dot = tf.matmul(u, self.context_vec)  # dot: (d, 1)
                return dot

            #def fn2(x):
            #    u = tf.tanh(tf.add(tf.matmul(x, weight1), bias1))  # u: (d, 50)
             #   dot = tf.matmul(u, self.context_vec2)  # dot: (d, 1)
             #   return dot

            outputs_time_major = tf.transpose(outputs, perm=[1, 0, 2])
            dots_time_major1 = tf.map_fn(fn1, outputs_time_major)
            #dots_time_major2 = tf.map_fn(fn2, outputs_time_major)
            self.dots = tf.transpose(dots_time_major1, perm=[1, 0, 2])
            #self.dots2 = tf.transpose(dots_time_major2, perm=[1, 0, 2])


            self.alphas = tf.nn.softmax(self.dots, dim=1) # alphas: (d, t, 1)
            #self.alphas2 = tf.nn.softmax(self.dots2, dim=1) # alphas: (d, t, 1)

            outputs_scaled = tf.multiply(self.alphas, outputs)
            #outputs_scaled2 = tf.multiply(self.alphas2, outputs)
            #normalize_a = tf.nn.l2_normalize(self.context_vec, 0)
            #normalize_b = tf.nn.l2_normalize(self.context_vec2, 0)
            #self.cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b))

            #self.context_weight1 = tf.Variable(tf.constant(1.0))
            #self.context_weight2 = tf.Variable(tf.constant(1.0))
            #output_final = tf.subtract(tf.multiply(self.context_weight1, outputs_scaled), tf.multiply(self.context_weight2, outputs_scaled2))
            #output_final = tf.multiply(self.context_weight1, outputs_scaled)
            #outputs_concat = tf.concat([outputs_scaled, outputs_scaled2], axis=-1)
            attention_outputs = tf.reduce_sum(outputs_scaled, axis=1) # v: (d, e)

        regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_scale)
        with tf.name_scope("dense1"):
            self.dense1 = tf.layers.dense(
                inputs=attention_outputs,
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
                inputs=dropout,
                units=num_classes,
                kernel_regularizer = regularizer
            )  # TODO: add initialization

        with tf.name_scope("loss"):
            xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            #self.loss = tf.add(tf.multiply(self.cos_similarity, 0.02), tf.reduce_mean(xentropy))
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
