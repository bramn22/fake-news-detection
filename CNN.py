import tensorflow as tf
import numpy as np

class CNN:
    ''' This class is very much based on https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py'''
    def __init__(self, seq_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters):
        self.x = tf.placeholder(tf.int32, [None, None], name="x") # None because of difference in training and eval data dimensions
        self.y = tf.placeholder(tf.float32, [None, num_classes], name="y") # TODO: why y: float and x: int??
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("embedding"):
            W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), trainable=False, name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
            self.embedding_init = W.assign(self.embedding_placeholder)
            self.vecs = tf.nn.embedding_lookup(W, self.x)
            self.vecs_expanded = tf.expand_dims(self.vecs, -1) # shape of [None, seq_length, embedding_size, 1]

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                conv = tf.layers.conv2d(
                    inputs=self.vecs_expanded,
                    filters=num_filters,
                    kernel_size=[filter_size, embedding_size],
                    padding="valid",
                    use_bias=True,
                    activation=tf.nn.relu
                )

                pool = tf.layers.max_pooling2d(
                    inputs=conv,
                    pool_size=[seq_length - filter_size + 1, 1],
                    strides=1,
                    padding="valid"
                )
                pooled_outputs.append(pool)

        num_filters_total = num_filters * len(filter_sizes)
        self.pool_concat = tf.concat(pooled_outputs, axis=3)
        self.pool_flat = tf.reshape(self.pool_concat, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.dropout = tf.nn.dropout(self.pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(
                inputs=self.dropout,
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
        learning_rate = 0.001
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(self.loss, global_step=global_step), global_step

