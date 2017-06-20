from .agent import Agent
import numpy as np
import tensorflow as tf
import random
import matplotlib
import scipy.signal
from helper import timeit
import os
from collections import Counter
from .hierarchical_experience import Hexperience
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class HierarchyAgent(Agent):
    def __init__(self,
                 sess,
                 conf,
                 pred_network_m,  # manager
                 pred_network_w,  # worker
                 target_network_m=None,
                 target_network_w=None,
                 name='H_Agent_'):
        super(HierarchyAgent, self).__init__(sess, conf, name)

        self.expand_size = conf.expand_size
        self.height_m = self.width_m = conf.size_manager
        self.height_w = self.width_w = conf.size_worker
        self.width = self.width_m * self.width_w
        self.height = self.height_m * self.height_w
        self.n_actions_m = self.width_m * self.height_m + 1
        self.n_actions_w = self.width_w * self.height_w
        assert self.width * self.height == self.n_features

        # mnist unit mask
        self.mnist_unit_mask = [self.unit_mask_create(i) for i in range(self.n_features)]
        self.mnist_unit_mask = np.array(self.mnist_unit_mask)

        self.pred_network_m = pred_network_m
        self.pred_network_w = pred_network_w
        self.target_network_m = target_network_m
        self.target_network_w = target_network_w
        self.double_q = conf.double_q
        self.target_network_m.create_copy_op(self.pred_network_m)
        self.target_network_w.create_copy_op(self.pred_network_w)
        with tf.variable_scope(self.name):
            self.build_training_tensor()
            self.build_classifier()

        observation_dim = self.input_dim - self.n_features
        self.experience = Hexperience(conf.batch_size, conf.memory_size,
                                      self.n_features, self.n_classes, [observation_dim])

    def mnist_expand(self, x):
        datum = np.zeros([28 * self.expand_size, 28 * self.expand_size])
        ind = np.random.randint(self.expand_size * self.expand_size)
        ind_row = ind // self.expand_size
        ind_col = ind % self.expand_size
        datum[(ind_row * 28):(ind_row * 28 + 28), (ind_col * 28):(ind_col * 28 + 28)] = x.reshape(28, 28)
        return datum

    def unit_mask_create(self, n):
        i = n // (4 * self.expand_size)
        j = n % (4 * self.expand_size)
        msk = np.zeros((28*self.expand_size, 28*self.expand_size),  dtype=np.uint8)
        msk[7 * i: 7 * i + 7, 7 * j: 7 * j + 7] = 1
        return msk

    def mnist_mask_batch_old(self, acquired):
        lst = []
        for i in range(acquired.shape[0]):
            ones = np.where(acquired[i])[0]
            msk = np.sum(self.mnist_unit_mask[ones], axis=0)
            lst.append(msk)
        return np.array(lst)

    def mnist_mask_batch(self, acquired):
        res = np.zeros([acquired.shape[0], 28 * self.expand_size, 28 * self.expand_size])  # might be changed to np.empty
        axis0, axis1 = np.where(acquired)
        res[axis0] = np.sum(self.mnist_unit_mask[axis1], axis=0)
        return res

    def get_observed(self, x, acquired):
        mnist_mask = self.mnist_mask_batch(acquired.reshape(1, -1)).reshape(-1)
        observed = x * mnist_mask
        return observed

    def build_training_tensor(self):
        # training tensor
        self.lr = tf.placeholder(tf.float32, shape=[])
        self.targets_m = tf.placeholder('float32', [None], name='target_q_t_m')
        self.actions_m = tf.placeholder('int64', [None], name='action_m')
        actions_one_hot_m = tf.one_hot(self.actions_m, self.n_actions_m)
        pred_q_m = tf.reduce_sum(self.pred_network_m.outputs * actions_one_hot_m,
                                 axis=1, name='q_acted_m')
        delta_m = self.targets_m - pred_q_m
        self.loss_m = tf.reduce_mean(tf.square(delta_m))
        optimizer_m = tf.train.RMSPropOptimizer(self.lr, momentum=0.95, epsilon=0.01)
        self.optim_m = optimizer_m.minimize(self.loss_m)

        self.targets_w = tf.placeholder('float32', [None], name='target_q_t_w')
        self.actions_w = tf.placeholder('int64', [None], name='action_w')
        actions_one_hot_w = tf.one_hot(self.actions_w, self.n_actions_w)
        pred_q_w = tf.reduce_sum(self.pred_network_w.outputs * actions_one_hot_w,
                                 axis=1, name='q_acted_w')
        delta_w = self.targets_w - pred_q_w
        self.loss_w = tf.reduce_mean(tf.square(delta_w))
        optimizer_w = tf.train.RMSPropOptimizer(self.lr, momentum=0.95, epsilon=0.01)
        self.optim_w = optimizer_w.minimize(self.loss_w)

    def choose_action(self, x, acquired, eps):
        observed = self.get_observed(x, acquired)
        inputs = np.concatenate((observed, acquired)).reshape(1, -1)

        weights = np.zeros(self.n_actions_m)
        weights[:-1] = scipy.signal.convolve2d(
            acquired.reshape(self.height, self.width),
            np.ones((self.height_w, self.width_w)),
            'valid')[::self.height_w, ::self.width_w].flatten()
        weights[-1] = 0  # classification
        masking = weights[:] # copy
        weights = self.height_w * self.width_w - weights
        weights[-1] = 1
        weights = weights / weights.sum()
        masking = masking // (self.height_w * self.width_w)
        # print(masking)
        # print(weights)
        masking = masking.reshape(1, -1)
        action_m, prob_m = self.pred_network_m.calc_actions(
                                            inputs,
                                            masking,
                                            policy='eps_greedy',
                                            eps=eps)
        action_m = action_m[0]
        action_w = -1  # For terminal action of manager
        if random.random() < eps:
            action_m = np.random.multinomial(1, weights, size=1)[0].argmax()
        if action_m != self.n_actions_m - 1:
            #Non-terminal action
            # observed_w = self.to_worker(action_m, observed, self.height, self.width)
            acquired_w = self.to_worker(action_m, acquired, self.height, self.width)
            # print('choose_action: shape of observed_w:', observed_w.shape)
            # print('choose_action: shape of acquired_w:', acquired_w.shape)
            # One-hot of manager's action
            onehot = np.zeros(self.n_actions_m - 1)
            onehot[action_m] = 1
            inputs_w = np.concatenate((observed, onehot, acquired)).reshape(1, -1)
            action_w, prob_w = self.pred_network_w.calc_actions(inputs_w,
                                                                acquired_w,
                                                                policy='eps_greedy',
                                                                eps=eps)
            action_w = action_w[0]
            if random.random() < eps:
                valid_actions = np.where(acquired_w.flatten() == 0)[0]
                action_w = random.choice(valid_actions)

        return action_m, action_w

    def train(self, tr_data, tr_labels, verbose=True):
        self.update_target_q_network()
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
                acquired = np.zeros(self.n_features)
                # add initial state to replay memory
                if random.random() > 0.5:
                    self.experience.add(np.zeros(x.shape), acquired, 0, -1, -1, False, label)
                n_acquired = 0
                epi_reward = 0
                terminal = False
                while not terminal:
                    # choose action
                    if total_steps < self.pre_train_steps:
                        action = self.random_action(acquired)
                    else:
                        action = self.choose_action(x, acquired, self.eps)
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
                        observed = self.get_observed(x, acquired)
                        prob, pred, correct = self.clf_predict(observed, acquired, label)
                        prob = prob.reshape(-1)
                        pred = pred[0]
                        correct = correct[0]
                        accuracy_history.append(int(correct))
                        assert len(acquired.shape) == 1
                        if correct:
                            sorted_prob = np.sort(prob)
                            reward = 10*(sorted_prob[-1] - sorted_prob[-2])
                        else:
                            reward = self.r_wrong
                    epi_reward += reward
                    # save experience
                    self.experience.add(observed, acquired, reward, action[0], action[1], terminal, label)
                    if terminal:
                        if n_acquired == 1 and verbose:
                            print("label", np.argmax(label))
                            print("prediction", pred)
                        break
                reward_history.append(epi_reward)
                assert n_acquired == np.sum(acquired)
                n_acquired_history.append(n_acquired)
                # sample batch
                prestates, unmissing_pre, actions_t_m, actions_t_w, rewards, poststates, unmissing, terminals, labels \
                    = self.experience.sample()
                s_t = np.concatenate((prestates, unmissing_pre), axis=1)
                s_t_plus_1 = np.concatenate((poststates, unmissing), axis=1)
                # TODO: check dimensions
                non_terminal_actions_t = np.where(actions_t_m != self.n_actions_m - 1)
                prestates_w = prestates[non_terminal_actions_t]
                unmissing_pre_w = unmissing_pre[non_terminal_actions_t]
                actions_t_m_w = actions_t_m[non_terminal_actions_t]
                actions_t_w = actions_t_w[non_terminal_actions_t]
                onehot = np.zeros((actions_t_m_w.size, self.n_actions_m - 1))
                onehot[np.arange(actions_t_m_w.size), actions_t_m_w] = 1
                s_w_t = np.concatenate((prestates_w,
                                        onehot,
                                        unmissing_pre_w),
                                       axis=1)
                targets_m, targets_w, non_terminal_actions_t_plus_1 = self.calc_targets(poststates, unmissing, labels)
                targets_w = targets_w[non_terminal_actions_t]
                targets_m = targets_m * np.where(terminals, 0, 1)
                targets_m[np.isnan(targets_m)] = 0
                targets_w[np.isnan(targets_w)] = 0
                targets_m = rewards + self.discount * targets_m  # Use same rewards for both targets
                targets_w = rewards[non_terminal_actions_t] + self.discount * targets_w

                # train
                clf_inputs = np.concatenate((s_t, s_t_plus_1), axis=0)
                clf_true_class = np.concatenate((labels, labels), axis=0)

                _, _, _, loss_m, loss_w, q_t_m, q_t_w, clf_accuracy, clf_softmax =\
                      self.sess.run([self.optim_m, self.optim_w,
                                     self.clf_optim,
                                     self.loss_m, self.loss_w,
                                     self.pred_network_m.outputs, self.pred_network_w.outputs,
                                     self.clf_accuracy,
                                     self.clf_softmax],
                          feed_dict={self.targets_m: targets_m,
                                     self.targets_w: targets_w,
                                     self.actions_m: actions_t_m,
                                     self.actions_w: actions_t_w,
                                     self.pred_network_m.inputs: s_t,
                                     self.pred_network_w.inputs: s_w_t,
                                     self.lr: lr,
                                     self.clf_inputs: clf_inputs,#
                                     self.true_class: clf_true_class,
                                     self.clf_lr: clf_lr})

                total_steps += 1
                if total_steps > self.pre_train_steps and self.eps > self.endE:
                    self.eps -= self.eps_decay

                # update target network parameter
                if total_steps % self.update_freq == self.update_freq - 1:
                    #print("Target Q update")
                    self.update_target_q_network()
                if (total_steps + 1) % 100 == 0:
                    print("------------------(", total_steps + 1, "/", \
                          self.n_epoch * self.train_size, ")------------------")
                    print("> current eps    :", self.eps)
                    print("> mean n_acquired:", np.mean(n_acquired_history))
                    print("> accuracy       :", np.mean(accuracy_history))
                    print("> reward         :", np.mean(reward_history))
                    history.append(np.mean(n_acquired_history))
                    n_acquired_history = []   # reset
                    accuracy_history = []
                    reward_history = []
        print("------------------ train done ------------------")
        saver = tf.train.Saver()
        saver.save(self.sess, self.save_path)

        return history

    def calc_targets(self, observation, acquired, labels):
        input_m = np.concatenate((observation, acquired), axis=1)
        # print("argmax_actions: shape of observation and acquired:", observation.shape, acquired.shape)
        ### double DQN
        acq = acquired.reshape(-1, self.height, self.width)
        sample_size = acq.shape[0]
        masking = np.zeros((sample_size, self.n_actions_m))
        for i in range(sample_size):
            masking[i][:-1] = scipy.signal.convolve2d(
                acquired[i].reshape(self.height, self.width),
                np.ones((self.height_w, self.width_w)),
                'valid')[::self.height_w, ::self.width_w].flatten() // (self.height_w * self.width_w)
            masking[i][-1] = 0  # classification
        # mask_m = np.concatenate((acquired, np.zeros((acquired.shape[0], 1))), axis=1)
        assert self.double_q
        # calc argmax_a Q_predict(s_{t+1}, a)
        actions_m, _ = self.pred_network_m.calc_actions(
            input_m,
            masking,
            eps=0,
            policy='eps_greedy')
        # calc Q_target(s_{t+1}, a_{t+1})
        terminal_actions = np.where(actions_m == self.n_actions_m-1)
        lbls = labels[terminal_actions]
        obs_term = observation[terminal_actions]
        acq_term = acquired[terminal_actions]
        prob, _, correct = self.clf_predict(obs_term, acq_term, lbls)
        sorted_prob = np.sort(prob)
        diff_prob = 10*(sorted_prob[:, -1] - sorted_prob[:, -2])
        reward_wrong = np.full(correct.shape, self.r_wrong)
        reward = np.where(correct, diff_prob, reward_wrong)
        non_terminal_actions = np.where(actions_m != self.n_actions_m-1)
        targets_m = self.target_network_m.calc_outputs_with_idx(
            input_m,
            np.column_stack((np.arange(actions_m.size), actions_m)))
        targets_w = np.zeros(targets_m.shape)
        targets_w[terminal_actions] = reward
        if non_terminal_actions[0].size == 0:
            return targets_m, targets_w, non_terminal_actions
        observation_w = observation[non_terminal_actions]
        acquired_w = acquired[non_terminal_actions]
        acts_m = actions_m[non_terminal_actions]
        mask_w = self.to_worker_vector(acts_m, acquired_w, self.height, self.width)
        onehot = np.zeros((acts_m.size, self.n_actions_m - 1))
        input_w = np.concatenate((observation_w, onehot, acquired_w), axis=1).reshape(-1, self.input_dim +
                                                                                      self.n_actions_m - 1)
        onehot[np.arange(acts_m.size), acts_m] = 1
        actions_w, _ = self.pred_network_w.calc_actions(input_w,
                                                        mask_w,
                                                        policy='eps_greedy',
                                                        eps=0)
        targets_w[non_terminal_actions] = self.target_network_w.calc_outputs_with_idx(
            input_w,
            np.column_stack((np.arange(actions_w.size), actions_w)))
        return targets_m, targets_w, non_terminal_actions


    # Given manager's action, return part of worker's data
    def to_worker(self, action_m, data, height, width):
        ind_row = action_m // self.width_m
        ind_col = action_m % self.width_m
        h_w = height // self.height_m  # 14
        w_w = width // self.width_m  # 14
        data_w = data.reshape(
            height,
            width)[(ind_row * h_w):(ind_row * h_w + h_w), (ind_col * w_w):(ind_col * w_w + w_w)].flatten()
        # onehot = np.zeros((1, self.n_actions_m - 1))
        # onehot[action_m] = 1
        return data_w

    # Given manager's actions, return part of worker's data
    # data should be of shape [batch, height, width]
    # TODO: Optimize
    def to_worker_vector(self, action_m, data, height, width):
        ind_row = action_m // self.width_m
        ind_col = action_m % self.width_m
        data = data.reshape((-1, height, width))
        h = height // self.height_m  # 14
        w = width // self.width_m  # 14
        lst = []
        for i in range(len(action_m)):
            lst.append(data[i][(ind_row[i] * h):(ind_row[i] * h + h),
                       (ind_col[i] * w):(ind_col[i] * w + w)].flatten())
        return np.array(lst)

    def update_target_q_network(self):
        assert self.target_network_m is not None
        assert self.target_network_w is not None
        self.target_network_m.run_copy()
        self.target_network_w.run_copy()

    def is_terminal(self, action):
        return action[0] == self.n_actions_m - 1

    def random_action(self, acquired):
        masking = np.zeros(self.n_actions_m)
        masking[:-1] = self.height_w * self.width_w - scipy.signal.convolve2d(
            acquired.reshape(self.height, self.width),
            np.ones((self.height_w, self.width_w)),
            'valid')[::self.height_w, ::self.width_w].flatten()
        # print(valid_actions.size)
        masking[-1] = 1  # classification
        # valid_actions = np.where(masking == 0)[0]
        # action_m = random.choice(valid_actions)
        masking = masking / masking.sum()
        action_m = np.random.multinomial(1, masking, size=1)[0].argmax()
        action_w = -1  # For terminal action of manager
        if action_m != self.n_actions_m - 1:
            #Non-terminal action
            acquired_w = self.to_worker(action_m, acquired, self.height, self.width)
            valid_actions = np.where(acquired_w.flatten() == 0)[0]
            action_w = random.choice(valid_actions)
        return action_m, action_w

    # Given actions, return index of "acquired" matrix
    def index(self, action):
        # Tested!
        y_m = action[0] // self.width_m
        x_m = action[0] % self.width_m
        y_w = action[1] // self.width_w
        x_w = action[1] % self.width_w
        return (y_m * self.height_w + y_w) * self.width + x_m * self.width_w + x_w

    def update_acquired(self, acquired, action):
        acq_idx = self.index(action)
        # print(action_m, action_w, acq_idx)
        if acquired[acq_idx] == 1:
            print(action)
            assert False
        acquired[acq_idx] = 1

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
            acquired = np.zeros(self.n_features)
            terminal = False
            while not terminal:# np.any(acquired==0):
                # act (feature acquisition)
                action_m, action_w = self.choose_action(datum, acquired, 0)
                action = self.choose_action(datum, acquired, 0)
                if action_m != self.n_actions_m - 1:
                    # feature acquisition
                    acq_idx = self.index([action_m, action_w])
                    assert acquired[acq_idx] != 1
                    acquired[acq_idx] = 1
                else:
                    terminal = True
                # print(acquired.reshape([8,8]))
                # input('press any key to continue')
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
        for i in range(1, self.n_features + 1):
            idx, = np.where(n_acquired_history == i)
            if idx.shape[0]:
                print("Accuracy(n_acquired={:02d}):".format(i), np.mean(acc_history[idx]))
        print("Mean n_acquired : ", mean_)
        print("Min n_acquired  : ", min_)
        print("Max n_acquired  : ", max_)
        print("Med n_acquired  : ", med_)
        print("Detail          : ", detail)
        return accuracy, mean_, min_, max_, med_, detail