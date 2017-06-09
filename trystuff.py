from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from helper import mnist_expand, mnist_mask_batch
import numpy as np

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
train_data = mnist.train
test_data = mnist.test
train_data_features = train_data.images
train_data_labels = train_data.labels
test_data_features = test_data.images
test_data_labels = test_data.labels
plt.imshow(train_data_features[0].reshape(28, 28))
plt.show()
a = mnist_expand(train_data_features[0], 2)
plt.imshow(a)
plt.show()
n_features = 16 * 4
acquired = np.zeros(n_features)
acquired[[0, 2, 4, 6]] = 1
print(acquired)
mask = mnist_mask_batch(acquired.reshape(1, -1), size=2)
plt.imshow(mask.reshape(28*2,28*2))
plt.show()
print(a)