import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

learning_rate = 0.03
epochs = 3000

def logistic_regression(data):
    print("-----shape------")
    print(data.x[0])
    print(data.x.shape)
    print(data.x[0].shape)
    print(data.y.shape)
    X, y, y_pred, cost = get_model(data.x.shape[1], data.y.shape[1])
    run(data, X, y, y_pred, cost)
    print("Not yet implemented")

def run(data, X, y, y_pred, cost):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        # feed_dict = {x: x_batch, y: y_batch}
        for _ in range(epochs):
            feed_dict = {X: data.x, y: data.y}
            loss_val, _ = session.run([cost, train_step], feed_dict)
            print('loss:', loss_val.mean())

        #y_pred_all = session.run(y_pred, {X: data.x})
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1)), dtype=tf.float32))
        accuracy_value = session.run(accuracy, feed_dict={X: data.x, y: data.y})
    print(accuracy_value)
    #plt.figure(1)
    #plt.scatter(data.x, data.x)
    #plt.plot(data.x, y_pred_all)
    #plt.show()

def get_model(num_features, num_classes):
    print('---n----')
    print(num_features)
    print(num_classes)
    X = tf.placeholder(tf.float32, shape=(None, num_features), name="X")
    y = tf.placeholder(tf.float32, shape=(None, num_classes), name="y")

    W = tf.Variable(np.zeros((num_features, num_classes)), name="W", dtype=tf.float32)
    b = tf.Variable(np.zeros(num_classes), name="b", dtype=tf.float32)
    # set y_pred to true or false
    y_pred = tf.nn.softmax(tf.add(tf.matmul(X, W), b))
    #cost = tf.reduce_mean(-y*tf.log(y_pred) - (1-y)*tf.log(1-y_pred))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
    #cost = tf.nn.l2_loss(y_pred - y, name="squared_error_cost")
    return X, y, y_pred, cost

