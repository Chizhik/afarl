from tensorflow.examples.tutorials.mnist import input_data
from helper import mnist_expand, mnist_mask_batch
from network.ops import conv2d
import matplotlib
import numpy as np
from network.ops import *
import tensorflow as tf
import os
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class MnistExpanded(object):
    def __init__(self, sess, name="expanded_clf"):
        self.sess = sess
        self.mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
        # train_data = mnist.train
        # test_data = mnist.test
        # self.train_data_features = train_data.images
        # self.train_data_labels = train_data.labels
        # self.test_data_features = test_data.images
        # self.test_data_labels = test_data.labels
        # # plt.imshow(self.train_data_features[0].reshape(28, 28))
        # plt.show()
        # a = mnist_expand(self.train_data_features[0], 2)
        # plt.imshow(a)
        # plt.show()
        # n_features = 16 * 4
        # acquired = np.zeros(n_features)
        # acquired[[0, 2, 4, 6]] = 1
        # print(acquired)
        # mask = mnist_mask_batch(acquired.reshape(1, -1), size=2)
        # plt.imshow(mask.reshape(28*2,28*2))
        # plt.show()
        # print(a)
        self.dim = [None, 56, 56, 1]
        self.learning_rate = 0.001
        self.training_iters = 200000
        self.batch_size = 128
        self.display_step = 10
        self.n_input = 3136  # MNIST data input (img shape: 28*28)
        self.n_classes = 10  # MNIST total classes (0-9 digits)
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_classes])
        self.build_classifier()
        # Initializing the variables
        self.init = tf.global_variables_initializer()
        self.name = name
        self.save_dir = os.path.dirname(__file__)
        self.save_dir = os.path.join(self.save_dir, "expand")
        self.save_path = os.path.join(self.save_dir, name + '.ckpt')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def build_classifier(self):
        # Store layers weight & bias
        weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 2])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 2, 4])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([56 * 56 * 4, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, self.n_classes]))
        }

        biases = {
            'bc1': tf.Variable(tf.random_normal([2])),
            'bc2': tf.Variable(tf.random_normal([4])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        # Construct model
        self.pred = conv_net(self.x, weights, biases)

        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred,
                                                                           labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def mnist_expand(self, x, size):
        shape = [x.shape[0], 28 * size, 28 * size]
        datum = np.zeros(shape)
        ind = np.random.randint(size * size, size=shape[0])
        ind_row = ind // size
        ind_col = ind % size
        for i in range(shape[0]):
            datum[i][(ind_row[i] * 28):(ind_row[i] * 28 + 28), (ind_col[i] * 28):(ind_col[i] * 28 + 28)] = x[i].reshape(28, 28)
        return datum.reshape(shape[0], -1)


    def train(self):
        self.sess.run(self.init)
        step = 1
        losses = []
        accs = []
        # Keep training until reach max iterations
        while step * self.batch_size < self.training_iters:
            batch_x, batch_y = self.mnist.train.next_batch(self.batch_size)
            batch_x = self.mnist_expand(batch_x, 2)
            # Run optimization op (backprop)
            self.sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y})
            if step % self.display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = self.sess.run([self.cost, self.accuracy], feed_dict={self.x: batch_x,
                                                                  self.y: batch_y})
                losses.append(loss)
                accs.append(acc)
                print("Iter " + str(step * self.batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")
        saver = tf.train.Saver()
        saver.save(self.sess, self.save_path)
        plt.plot(losses)
        plt.savefig('losses' + time.strftime("%Y-%m-%d-%I:%M", time.localtime()) + '.png')

    def test(self):
        ckpt = tf.train.get_checkpoint_state(self.save_dir)
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.save_dir, ckpt_name)
            saver.restore(self.sess, fname)
            print(" [*] Load SUCCESS: %s" % fname)
        else:
            print(" [!] Load FAILED: %s")
            return False
        x = self.mnist_expand(self.mnist.test.images[:128], 2)
        y = self.mnist_expand(self.mnist.test.labels[:128], 2)
        print("Testing Accuracy:", self.sess.run(self.accuracy,
                                                 feed_dict={self.x: x,
                                                            self.y: y}))

with tf.Session() as sess:
    a = MnistExpanded(sess)
    a.train()
    a.test()
