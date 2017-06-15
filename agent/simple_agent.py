from .agent import Agent
import numpy as np
import tensorflow as tf
import random
import matplotlib
from helper import timeit
import os
from collections import Counter
from .experience import Experience
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SimpleAgent(Agent):
    def __init__(self,
                 sess,
                 conf,
                 pred_network,
                 target_network=None,
                 name='SimpleAgent'):
        super(SimpleAgent, self).__init__(sess, conf, name)

        self.n_actions = self.n_features + 1
        self.expand_size = conf.expand_size

        # mnist unit mask
        self.mnist_unit_mask = [self.unit_mask_create(i) for i in range(self.n_features)]
        self.mnist_unit_mask = np.array(self.mnist_unit_mask)
        print(self.mnist_unit_mask.shape)

        # network
        self.double_q = conf.double_q
        self.pred_network = pred_network
        self.target_network = target_network
        self.target_network.create_copy_op(self.pred_network)
        with tf.variable_scope(self.name):
            self.build_training_tensor()
            self.build_classifier()

        # experience replay memory init
        observation_dim = self.input_dim - self.n_features
        self.experience = Experience(conf.batch_size, conf.memory_size,
                                     self.n_features, self.n_classes, [observation_dim])
        # # # [self.n_features] should be changed! to real observation dim

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
        self.optim = optimizer.minimize(self.loss)

    @timeit
    def train(self, tr_data, tr_labels, verbose=True):
        self.update_target_q_network()
        summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
        total_steps = 0
        # total_steps = self.pre_train_steps # remove after debugging
        # self.eps = 0.2 # remove after debugging
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
                    # self.print_mask(x, acquired)
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
                s_t = np.concatenate((prestates, unmissing_pre), axis=1)
                s_t_plus_1 = np.concatenate((poststates, unmissing), axis=1)
                targets = self.calc_targets(unmissing, s_t_plus_1, poststates, terminals, rewards)
                # train
                clf_inputs = np.concatenate((s_t, s_t_plus_1), axis=0)
                clf_true_class = np.concatenate((labels, labels), axis=0)

                _, _, loss, q_t, clf_accuracy, clf_softmax = self.train_sess_run(targets, actions_t, s_t, lr,
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
        return self.sess.run([self.optim, self.clf_optim,
                              self.loss,
                              self.pred_network.outputs,
                              self.clf_accuracy,
                              self.clf_softmax],
                             feed_dict={self.targets: targets,
                                        self.actions: actions_t,
                                        self.pred_network.inputs: s_t,
                                        self.lr: lr,
                                        self.clf_inputs: clf_inputs,
                                        self.true_class: clf_true_class,
                                        self.clf_lr: clf_lr})

    def calc_targets(self, unmissing, s_t_plus_1, poststates, terminals, rewards):
        ### double DQN
        mask = np.concatenate((unmissing, np.zeros((unmissing.shape[0], 1))), axis=1)
        if self.double_q:
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
        ###### DQN
        else:
            # calc max Q_targets(s_{t+1}, a_{t+1})
            targets = self.target_network.calc_max_outputs_with_acquired(
                poststates, mask)
            # targets = self.target_network.calc_max_outputs(s_t_plus_1)
        # calc target value
        targets = targets * np.where(terminals, 0, 1)
        targets[np.isnan(targets)] = 0  # <------------ what is this?? 0 x inf
        return rewards + self.discount * targets

    def random_action(self, acquired):
        possible_actions = np.zeros(self.n_actions)
        possible_actions[:-1] = acquired
        indices = np.where(possible_actions == 0)[0]
        # print('Random Action')
        # print(acquired.reshape([8, 8]))
        # print(indices)
        # print('---------------------------------------------------')
        return random.choice(indices)

    def choose_action(self, x, acquired, eps, policy='eps_greedy'):
        observed = self.get_observed(x, acquired)
        inputs = np.concatenate((observed, acquired)).reshape(1, -1)
        masking = np.zeros((1, len(acquired)+1))
        masking[0, :self.n_features] = acquired
        masking[0, self.n_features] = 0 # making decision action
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
            # print("Epsilon greedy")
            # print('Choose Action')
            # print(acquired.reshape([8, 8]))
            if random.random() < eps:
                # print('eps greedy random')
                #action = random.choice(range(self.n_actions))
                missing = np.where(masking == 0)[1]
                # missing = [i for i, v in enumerate(acquired) if v==0]
                # missing.append(self.n_actions - 1)
                action = random.choice(missing)
                # print(action)
                # print('---------------------------------------------------')


        return action

    def get_observed(self, x, acquired):
        if self.data_type == 'mnist':
            mnist_mask = self.mnist_mask_batch(acquired.reshape(1, -1)).reshape(-1)
            observed = x * mnist_mask
        else:
            observed = x * acquired
        return observed

    def update_acquired(self, acquired, action):
        assert acquired[action] != 1
        acquired[action] = 1

    def is_terminal(self, action):
        return action == self.n_actions - 1

    def update_target_q_network(self):
        assert self.target_network is not None
        self.target_network.run_copy()

    def print_mask(self, x, acquired):
        print(acquired.reshape([8, 8]))
        # plt.figure()
        # plt.imshow(x.reshape(56, 56))
        # plt.show()
        # mnist_mask_ta = mnist_mask_batch(acquired.reshape(1, -1), self.expand_size)
        # plt.imshow(mnist_mask_ta.reshape(56, 56))
        # plt.show()
        # mnist_mask = self.mnist_mask_batch_old(acquired.reshape(1, -1))
        # plt.imshow(mnist_mask.reshape(56, 56))
        # plt.show()
        mnist_mask2 = self.mnist_mask_batch(acquired.reshape(1, -1))
        plt.imshow(mnist_mask2.reshape(56, 56))
        plt.show()
        # obs = self.get_observed(x, acquired)
        # plt.imshow(obs.reshape(56, 56))
        # plt.show()


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
            plt.imshow(datum.reshape(56, 56))
            plt.savefig('test_datum.png')
            acquired = np.zeros(self.n_features)
            terminal = False
            while not terminal:# np.any(acquired==0):
                # act (feature acquisition)
                action = self.choose_action(datum, acquired, 0)
                if action < self.n_features:
                    assert acquired[action] != 1
                    acquired[action] = 1
                else:
                    terminal=True
                print(acquired.reshape([8,8]))
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