from network.layer import *
from network.mlp import MLPSmall
from .experience import Experience
from collections import Counter
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import tensorflow as tf
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('.'))))

# Some thoughts: Since both agent and classifier will train the cnn layers, we need to pay attention to their
# learning rates

class CNNAgent(object):
    def __init__(self,
                 sess,
                 conf,
                 name='Agent'):

        self.sess = sess
        self.var = {}
        # problem
        self.input_dim = conf.input_dim  # for example, [56, 56]
        self.expand_size = conf.expand_size
        # TODO: change conf.n_features to conf.feature_dim
        self.feature_dim = conf.n_features
        self.n_actions = self.feature_dim + 1
        self.n_classes = conf.n_classes
        self.train_size = conf.train_size
        self.test_size = conf.test_size
        self.data_type = conf.data_type
        # train Q
        self.update_freq = conf.target_update_freq
        self.discount = conf.discount
        self.policy = conf.policy
        self.max_lr = conf.max_lr # <----------------------------- to be changed
        self.min_lr = conf.min_lr
        self.clf_max_lr = conf.clf_max_lr
        self.clf_min_lr = conf.clf_min_lr
        self.batch_size = conf.batch_size
        self.n_epoch = conf.n_epoch
        self.pre_train_steps = conf.pre_train_steps
        self.startE = conf.eps_start
        self.endE = conf.eps_end
        self.anneling_steps = conf.anneling_steps
        self.eps_decay = (self.startE - self.endE) / self.anneling_steps
        self.eps = self.startE
        # reward
        self.r_cost = conf.r_cost
        self.r_wrong = conf.r_wrong
        self.r_repeat = conf.r_repeat
        self.r_correct = conf.r_correct
        # classfier
        self.clf_hidden_sizes = conf.clf_hidden_sizes
        # Q learning network
        self.double_q = conf.double_q
        self.Q_hidden_sizes = conf.Q_hidden_sizes
        # for model saving and restoring
        self.name = name
        self.save_dir = os.path.relpath(os.path.dirname(__file__))
        self.save_dir = os.path.join(self.save_dir, conf.save_dir)
        self.log_path = self.save_dir
        self.save_path = os.path.join(self.save_dir, name + '.ckpt')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # Experience Replay
        observation_dim = self.input_dim[0] * self.input_dim[1]
        self.experience = Experience(conf.batch_size, conf.memory_size,
                                     self.feature_dim, self.n_classes, [observation_dim])

        # mnist unit mask
        self.mnist_unit_mask = [self.unit_mask_create(i) for i in range(self.feature_dim)]
        self.mnist_unit_mask = np.array(self.mnist_unit_mask)
        print(self.mnist_unit_mask.shape)

        # Building stuff
        with tf.variable_scope(self.name):
            self.build_conv()
            self.build_classifier()
            self.build_agent()
            self.build_training_tensor()
        self.target_network.create_copy_op(self.pred_network)

    def mnist_expand(self, x):
        datum = np.zeros([self.input_dim[0], self.input_dim[1]])
        ind = np.random.randint(self.expand_size * self.expand_size)
        ind_row = ind // self.expand_size
        ind_col = ind % self.expand_size
        datum[(ind_row * 28):(ind_row * 28 + 28), (ind_col * 28):(ind_col * 28 + 28)] = x.reshape(28, 28)
        return datum

    def mnist_mask_batch(self, acquired):
        res = np.zeros([acquired.shape[0], self.input_dim[0], self.input_dim[1]])  # might be changed to np.empty
        axis0, axis1 = np.where(acquired)
        res[axis0] = np.sum(self.mnist_unit_mask[axis1], axis=0)
        return res

    def unit_mask_create(self, n):
        i = n // (4 * self.expand_size)
        j = n % (4 * self.expand_size)
        msk = np.zeros((28*self.expand_size, 28*self.expand_size),  dtype=np.uint8)
        msk[7 * i: 7 * i + 7, 7 * j: 7 * j + 7] = 1
        return msk

    def build_conv(self,
                   weights_initializer=initializers.xavier_initializer(),
                   biases_initializer=tf.constant_initializer(0.1),
                   hidden_activation_fn=tf.nn.relu):
        # Dimensions of input: [N, 28, 28, 2] or [N, 56, 56, 2] or ...
        self.conv_inputs = tf.placeholder('float32', [None,
                                                     self.input_dim[0],
                                                     self.input_dim[1],
                                                     2],
                                         name='conv_inputs')
        print('Shape of conv_inputs: ', self.conv_inputs.get_shape())
        with tf.variable_scope(self.name):
            self.conv_l1, self.var['conv_l1_w'], self.var['conv_l1_b'] = conv2d(self.conv_inputs,  # input of cnn
                                                                                32,  # number of out channels
                                                                                [8, 8],  # size of kernel (filter)
                                                                                [4, 4],  # strides
                                                                                weights_initializer,
                                                                                biases_initializer,
                                                                                hidden_activation_fn,
                                                                                'NHWC', name='conv_l1')
            print('Shape of conv_l1: ', self.conv_l1.get_shape())
            # it should be [None, 13, 13, 32]
            self.conv_l2, self.var['conv_l2_w'], self.var['conv_l2_b'] = conv2d(self.conv_l1,  # input of cnn
                                                                                64,  # number of out channels
                                                                                [4, 4],  # size of kernel (filter)
                                                                                [2, 2],  # strides
                                                                                weights_initializer,
                                                                                biases_initializer,
                                                                                hidden_activation_fn,
                                                                                'NHWC', name='conv_l2')
            print('Shape of conv_l2: ', self.conv_l2.get_shape())
            # it should be [None, 5, 5, 64]
            self.conv_l3, self.var['conv_l3_w'], self.var['conv_l3_b'] = conv2d(self.conv_l2,  # input of cnn
                                                                                64,  # number of out channels
                                                                                [3, 3],  # size of kernel (filter)
                                                                                [1, 1],  # strides
                                                                                weights_initializer,
                                                                                biases_initializer,
                                                                                hidden_activation_fn,
                                                                                'NHWC', name='conv_l3')
            print('Shape of conv_l3: ', self.conv_l3.get_shape())
            # it should be [None, 3, 3, 64]
        self.conv_out = tf.contrib.layers.flatten(self.conv_l3)
        self.conv_out_dim = self.conv_out.get_shape()[1]
        print('Out dimension of conv: ', self.conv_out_dim)

    # Make sure to call this method after calling build_conv
    def build_classifier(self):
        layer = self.conv_out
        self.true_class = tf.placeholder('float32', [None, self.n_classes],
            name='true_class')
        for idx, hidden_size in enumerate(self.clf_hidden_sizes):
          w_name, b_name = 'w_c_%d' % idx, 'b_c_%d' % idx
          layer, self.var[w_name], self.var[b_name] = \
                    linear(layer, hidden_size, activation_fn=tf.nn.relu,
                        biases_initializer=tf.constant_initializer(0.1),
                        name='lin_c_%d' % idx)
        self.clf_logits, self.var['w_c_out'], self.var['b_c_out'] = \
                    linear(layer, self.n_classes, activation_fn=None,
                            biases_initializer=tf.zeros_initializer(), name='lin_c_out')
        self.clf_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=self.clf_logits,
                        labels=self.true_class))
        self.clf_softmax = tf.nn.softmax(self.clf_logits)
        self.clf_pred = tf.argmax(self.clf_softmax, 1)
        self.clf_correct = tf.equal(tf.argmax(self.true_class, 1), self.clf_pred)
        self.clf_accuracy = tf.reduce_mean(tf.cast(self.clf_correct, 'float'))
        self.clf_lr = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.AdamOptimizer(learning_rate=self.clf_lr)
        self.clf_optim = optimizer.minimize(self.clf_loss)

    # inputs and acquired should be flattened
    def clf_predict(self, inputs, acquired, labels):
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        if len(acquired.shape) == 1:
            acquired = acquired.reshape(1, -1)
        if len(labels.shape) == 1:
            labels = labels.reshape(1, -1)
        mnist_mask = self.mnist_mask_batch(acquired).reshape([-1, self.input_dim[0], self.input_dim[1]])
        inputs = inputs.reshape([-1, self.input_dim[0], self.input_dim[1]])
        # mnist_mask and inputs now are of shape (example) [64, 56, 56] we need to stack them into shape [64, 56, 56, 2]
        conv_in = np.stack([inputs, mnist_mask], axis=3)

        return self.sess.run([self.clf_softmax, self.clf_pred, self.clf_correct],
                             feed_dict={
                                 self.conv_inputs: conv_in,
                                 self.true_class: labels})

    def build_agent(self):
        bias_init = tf.constant_initializer(0.1)
        self.pred_network = MLPSmall(sess=self.sess,
                                     observation_dims=self.conv_out_dim,  # but we actually don't need this
                                     output_size=self.n_actions,
                                     network_output_type='normal',
                                     biases_initializer=bias_init,
                                     hidden_sizes=self.Q_hidden_sizes,
                                     name='pred_network', trainable=True,
                                     input_tensor=self.conv_out,
                                     conv_input_placeholder=self.conv_inputs)
        self.target_network = MLPSmall(sess=self.sess,
                                       observation_dims=self.conv_out_dim,  # but we actually don't need this
                                       output_size=self.n_actions,
                                       network_output_type='normal',
                                       biases_initializer=bias_init,
                                       hidden_sizes=self.Q_hidden_sizes,
                                       name='target_network', trainable=False,
                                       input_tensor=self.conv_out,
                                       conv_input_placeholder=self.conv_inputs)

    def build_training_tensor(self):
        # training tensor
        self.lr = tf.placeholder(tf.float32, shape=[])
        self.targets = tf.placeholder('float32', [None], name='target_q_t')
        self.actions = tf.placeholder('int64', [None], name='action')
        actions_one_hot = tf.one_hot(self.actions, self.n_actions)
        pred_q = tf.reduce_sum(self.pred_network.outputs * actions_one_hot,
                               axis=1, name='q_acted')
        delta = self.targets - pred_q
        self.loss = tf.reduce_mean(tf.square(delta))
        optimizer = tf.train.RMSPropOptimizer(self.lr, momentum=0.95, epsilon=0.01)
        self.optim = optimizer.minimize(self.loss, var_list=list(self.pred_network.var.values()))

    # TODO: change inputs
    def train(self, tr_data, tr_labels, verbose=True):
        self.update_target_q_network()
        summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
        total_steps = 0
        n_acquired_history = []
        accuracy_history = []
        history = []
        reward_history = []
        lr = self.max_lr
        clf_lr = self.clf_max_lr
        for epoch in range(self.n_epoch):
            for x, label in zip(tr_data, tr_labels):
                x = self.mnist_expand(x).ravel()
                acquired = np.zeros(self.feature_dim)
                # add initial state to replay memory
                if random.random() > 0.5:
                    self.experience.add(np.zeros(x.shape), acquired, 0, -1, False, label)
                n_acquired = 0
                epi_reward = 0
                terminal = False
                while not terminal:
                    # choose action
                    if total_steps < self.pre_train_steps:
                        action = self.random_action(acquired)
                    else:
                        action = self.choose_action(x, acquired, self.eps, policy=self.policy)
                    if not self.is_terminal(action):
                        # feature acquisition
                        terminal = False
                        self.update_acquired(acquired, action)
                        reward = self.r_cost
                        n_acquired += 1
                        observed = self.get_observed(x, acquired)
                    else:
                        # make a decision (terminal state)
                        terminal = True
                        # TODO: combine following two call to optimize them
                        # Right now each of them call function mnist_mask_batch
                        observed = self.get_observed(x, acquired)
                        prob, pred, correct = self.clf_predict(observed, acquired, label)
                        prob = prob.reshape(-1)
                        pred = pred[0]
                        correct = correct[0]
                        accuracy_history.append(int(correct))
                        assert len(acquired.shape) == 1
                        if correct:
                            sorted_prob = np.sort(prob)
                            reward = sorted_prob[-1] - sorted_prob[-2]
                        else:
                            reward = self.r_wrong
                    epi_reward += reward
                    # save experience
                    self.experience.add(observed, acquired, reward, action, terminal, label)
                    if terminal:
                        if n_acquired == 1 and verbose:
                            print("label", np.argmax(label))
                            print("prediction", pred)
                        break
                reward_history.append(epi_reward)
                assert n_acquired == np.sum(acquired)
                n_acquired_history.append(n_acquired)
                # sample batch
                prestates, unmissing_pre, actions_t, rewards, poststates, unmissing, terminals, labels \
                    = self.experience.sample()
                s_t = self.get_conv_input_from_obs(prestates, unmissing_pre)
                s_t_plus_1 = self.get_conv_input_from_obs(poststates, unmissing)
                targets = self.calc_targets(unmissing, s_t_plus_1, poststates, terminals, rewards)
                # train
                clf_inputs = np.concatenate((s_t, s_t_plus_1), axis=0)
                clf_true_class = np.concatenate((labels, labels), axis=0)

                loss, q_t, clf_accuracy, clf_softmax = self.train_sess_run(targets, actions_t, s_t, lr,
                                                                           clf_inputs, clf_true_class,
                                                                           clf_lr)

                total_steps += 1
                if total_steps > self.pre_train_steps and self.eps > self.endE:
                    self.eps -= self.eps_decay

                # update target network parameter
                if total_steps % self.update_freq == self.update_freq - 1:
                    # print("Target Q update")
                    self.update_target_q_network()
                if (total_steps + 1) % 100 == 0:
                    print("------------------(", total_steps + 1, "/",
                          self.n_epoch * self.train_size, ")------------------")
                    print("> current eps    :", self.eps)
                    print("> mean n_acquired:", np.mean(n_acquired_history))
                    print("> accuracy       :", np.mean(accuracy_history))
                    print("> reward         :", np.mean(reward_history))
                    history.append(np.mean(n_acquired_history))
                    n_acquired_history = []  # reset
                    accuracy_history = []
                    reward_history = []
        print("------------------ train done ------------------")
        saver = tf.train.Saver()
        saver.save(self.sess, self.save_path)
        return history

    def train_sess_run(self, targets, actions_t, s_t, lr, clf_inputs, clf_true_class, clf_lr):
        # optimaze Q net
        _, loss, q_t = self.sess.run([self.optim, self.loss, self.pred_network.outputs],
                                     feed_dict={self.targets: targets,
                                                self.actions: actions_t,
                                                self.conv_inputs: s_t,
                                                self.lr: lr})
        # optimize classifier
        _, clf_accuracy, clf_softmax = self.sess.run([self.clf_optim, self.clf_accuracy, self.clf_softmax],
                                                     feed_dict={self.conv_inputs: clf_inputs,
                                                                self.true_class: clf_true_class,
                                                                self.clf_lr: clf_lr})

        return loss, q_t, clf_accuracy, clf_softmax

    def calc_targets(self, unmissing, s_t_plus_1, poststates, terminals, rewards):
        ### double DQN
        mask = np.concatenate((unmissing, np.zeros((unmissing.shape[0], 1))), axis=1)
        assert self.double_q
        # calc argmax_a Q_predict(s_{t+1}, a)
        actions_t_plus_1, _ = self.pred_network.calc_actions(
            s_t_plus_1,
            mask,
            eps=0,
            policy='eps_greedy')
        # calc Q_target(s_{t+1}, a_{t+1})
        targets = self.target_network.calc_outputs_with_idx(
            s_t_plus_1,
            [[idx, pred_a] for idx, pred_a in enumerate(actions_t_plus_1)])
        # calc target value
        targets = targets * np.where(terminals, 0, 1)
        targets[np.isnan(targets)] = 0  # <------------ what is this?? 0 x inf
        return rewards + self.discount * targets

    def update_target_q_network(self):
        assert self.target_network is not None
        self.target_network.run_copy()

    def random_action(self, acquired):
        possible_actions = np.zeros(self.n_actions)
        possible_actions[:-1] = acquired
        indices = np.where(possible_actions == 0)[0]
        return random.choice(indices)

    def choose_action(self, x, acquired, eps, policy='eps_greedy'):
        inputs = self.get_conv_input_from_x(x, acquired)
        masking = np.zeros((1, len(acquired)+1))
        masking[0, :self.feature_dim] = acquired
        masking[0, self.feature_dim] = 0 # making decision action
        action, prob = self.pred_network.calc_actions(
                                            inputs,
                                            masking,
                                            policy=policy,
                                            eps=eps)
        if policy=="softmax":
            prob = prob[0]
            action = np.random.multinomial(1, prob, size=1)[0].argmax()
        elif policy == 'eps_greedy':
            action = action[0]
            if random.random() < eps:
                #action = random.choice(range(self.n_actions))
                missing = np.where(masking == 0)[1]
                # missing = [i for i, v in enumerate(acquired) if v==0]
                # missing.append(self.n_actions - 1)
                action = random.choice(missing)

        return action

    def get_observed(self, x, acquired):
        if len(acquired.shape) == 1:
            acquired = acquired.reshape(1, -1)
        mnist_mask = self.mnist_mask_batch(acquired).reshape([-1, self.input_dim[0], self.input_dim[1]])
        x = x.reshape([-1, self.input_dim[0], self.input_dim[1]])
        observed = x * mnist_mask
        return observed.ravel()

    def get_conv_input_from_obs(self, datum, acquired):
        if len(acquired.shape) == 1:
            acquired = acquired.reshape(1, -1)
        mnist_mask = self.mnist_mask_batch(acquired).reshape([-1, self.input_dim[0], self.input_dim[1]])
        datum = datum.reshape([-1, self.input_dim[0], self.input_dim[1]])
        # mnist_mask and inputs now are of shape (example) [64, 56, 56] we need to stack them into shape [64, 56, 56, 2]
        return np.stack([datum, mnist_mask], axis=3)

    def get_conv_input_from_x(self, x, acquired):
        if len(acquired.shape) == 1:
            acquired = acquired.reshape(1, -1)
        mnist_mask = self.mnist_mask_batch(acquired).reshape([-1, self.input_dim[0], self.input_dim[1]])
        x = x.reshape([-1, self.input_dim[0], self.input_dim[1]])
        observed = x * mnist_mask
        return np.stack([observed, mnist_mask], axis=3)

    def update_acquired(self, acquired, action):
        assert acquired[action] != 1
        acquired[action] = 1

    def is_terminal(self, action):
        return action == self.n_actions - 1

    def stat(self, array):
        return np.mean(array), np.amin(array), np.amax(array), np.median(array)

    def test(self, data, labels, verbose=False):
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
        acc_history = []
        n_acquired_history = []
        for datum, label in zip(data, labels):
            datum = self.mnist_expand(datum).ravel()
            acquired = np.zeros(self.feature_dim)
            terminal = False
            while not terminal:# np.any(acquired==0):
                # act (feature acquisition)
                action = self.choose_action(datum, acquired, 0)
                if action < self.feature_dim:
                    assert acquired[action] != 1
                    acquired[action] = 1
                else:
                    terminal=True
            observed = self.get_observed(datum, acquired)
            prob, pred, correct = self.clf_predict(observed, acquired, label)
            prob = prob[0]
            pred = pred[0]
            correct = correct[0]

            sorted_prob = np.sort(prob)
            diff = sorted_prob[-1] - sorted_prob[-2]
            n_acquired = np.sum(acquired)
            if verbose:
                print("acquired feature num", n_acquired)
                print("Diff of first & second largest prob", diff)
                input()
            acc_history.append(int(correct))
            n_acquired_history.append(np.sum(acquired))

        accuracy = np.mean(acc_history)
        mean_, min_, max_, med_ = self.stat(n_acquired_history)
        detail = dict(Counter(n_acquired_history))
        acc_history = np.array(acc_history)
        n_acquired_history = np.array(n_acquired_history)
        assert acc_history.shape == n_acquired_history.shape
        print("Accuracy        : ", accuracy)
        for i in range(1, self.feature_dim + 1):
            idx, = np.where(n_acquired_history == i)
            if idx.shape[0]:
                print("Accuracy(n_acquired={:02d}):".format(i), np.mean(acc_history[idx]))
        print("Mean n_acquired : ", mean_)
        print("Min n_acquired  : ", min_)
        print("Max n_acquired  : ", max_)
        print("Med n_acquired  : ", med_)
        print("Detail          : ", detail)
        return accuracy, mean_, min_, max_, med_, detail

    def mnist_expand_batch(self, x):
        shape = [x.shape[0], 28 * self.expand_size, 28 * self.expand_size]
        datum = np.zeros(shape)
        ind = np.random.randint(self.expand_size * self.expand_size, size=shape[0])
        ind_row = ind // self.expand_size
        ind_col = ind % self.expand_size
        for i in range(shape[0]):
            datum[i][(ind_row[i] * 28):(ind_row[i] * 28 + 28), (ind_col[i] * 28):(ind_col[i] * 28 + 28)] = x[i].reshape(28, 28)
        return datum.reshape(shape[0], -1)

    def pretrain_clf(self, batch_size=64, training_iters=20000):
        mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
        step = 1
        accs = []
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = self.mnist_expand_batch(batch_x)
            acquired = np.ones([batch_size, self.feature_dim])
            inputs = self.get_conv_input_from_x(batch_x, acquired)
            # Run optimization op (backprop)
            # optimize classifier
            _, clf_accuracy, clf_softmax = self.sess.run([self.clf_optim, self.clf_accuracy, self.clf_softmax],
                                                         feed_dict={self.conv_inputs: inputs,
                                                                    self.true_class: batch_y,
                                                                    self.clf_lr: self.clf_max_lr})
            if step % 10 == 0:
                # Calculate batch loss and accuracy
                accs.append(clf_accuracy)
                print("Iter " + str(step * batch_size) + ", Training Accuracy= " + \
                      "{:.5f}".format(clf_accuracy))
            step += 1
        print("Optimization Finished!")