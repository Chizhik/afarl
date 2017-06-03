"""Modification of https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import random
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class Experience(object):
  def __init__(self, batch_size, memory_size, n_features, n_classes, observation_dims=[15]):
    #self.data_format = data_format
    self.batch_size = batch_size
    self.memory_size = memory_size
    self.n_features = n_features
    self.n_classes = n_classes
    self.observations = np.empty([self.memory_size] + observation_dims,
                                dtype=np.float)
    self.unmissing = np.empty([self.memory_size, self.n_features],
                                dtype=np.uint8)
    self.actions = np.empty(self.memory_size, dtype=np.uint8)
    self.rewards = np.empty(self.memory_size, dtype=np.float)
    self.terminals = np.empty(self.memory_size, dtype=np.bool)
    self.labels = np.empty([self.memory_size, self.n_classes])
    # pre-allocate prestates and poststates for minibatch
    self.prestates = np.empty([self.batch_size] + observation_dims, dtype = np.float16)
    self.poststates = np.empty([self.batch_size] + observation_dims, dtype = np.float16)

    self.count = 0
    self.current = 0

  def add(self, observed, acquired, reward, action, terminal, label):
    '''
        acquired_feat : mask form
        action : idx of action chosen
        terminal : terminal or not!
    '''

    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.observations[self.current, ...] = observed
    self.terminals[self.current] = terminal
    self.unmissing[self.current] = acquired
    self.labels[self.current] = label
    self.count = max(self.count, self.current + 1) # How many data?
    self.current = (self.current + 1) % self.memory_size # Current position

  def sample(self):
    indices = []
    while len(indices) < self.batch_size:
      while True:
        index = random.randint(0, self.count - 1)
        #index = random.randint(1, self.count - 1)
        if index == self.current:
            continue
        if self.terminals[(index - 1) % self.count]:
            continue
        break

      self.prestates[len(indices), ...] = self.retreive(index - 1)
      self.poststates[len(indices), ...] = self.retreive(index)
      indices.append(index)
    indices = np.array(indices)
    actions = self.actions[indices]
    rewards = self.rewards[indices]
    terminals = self.terminals[indices]
    unmissing = self.unmissing[indices]
    unmissing_pre = self.unmissing[(indices - 1) % self.count]
    labels = self.labels[indices]
    return self.prestates, unmissing_pre, actions, rewards,\
            self.poststates, unmissing, terminals, labels

  def retreive(self, index):
    index = index % self.count
    return self.observations[index, ...]
