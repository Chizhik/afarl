import numpy as np
import tensorflow as tf
from collections import Counter
from functools import reduce
import time

###############################
#          Decorator          #
###############################

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('{} {:2.2f} sec'.format(method.__name__, te - ts))
        return result

    return timed


def timeit_return(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('{} {:2.2f} sec'.format(method.__name__, te - ts))
        return result, te - ts

    return timed


def g(n):
    assert np.any(np.array(range(16)) == n)
    i = int(n // 4)
    j = int(n % 4)
    msk = np.zeros((28, 28), dtype=np.uint8)
    msk[7*i: 7*i + 7, 7*j: 7*j + 7] = 1
    msk = msk.reshape(-1)
    a, = np.where(msk==1)
    return a

mnist_feature_where = [g(i) for i in range(16)]

def mnist_mask_batch(acquired, size=1):
    mask = np.zeros((acquired.shape[0], 784*size*size))
    if np.all(acquired == 0):
        return mask
    axis0, axis1 = np.where(acquired)
    cnt = Counter(axis0)
    def f_mnist_axis0(x, y):
        if type(x) is not list:
            x = [x] * cnt[x] * 49
        return x + [y] * cnt[y] * 49

    def f_mnist_axis1(x, y):
        if type(x) is not np.ndarray:
            x = mnist_feature_where[x]
        return np.concatenate((x, mnist_feature_where[y]))
    axis0 = reduce(lambda x, y: f_mnist_axis0(x, y), np.sort(list(cnt)))
    axis1 = reduce(lambda x, y: f_mnist_axis1(x, y), axis1)

    mask[axis0, axis1] = 1
    return mask


@timeit
def data2seq(data, n_features, acquired=None, data_type="cube"):
    if not np.shape(acquired):
        acquired = np.ones((data.shape[0], n_features))
    mask = np.zeros((data.shape[0], n_features))
    axis0, axis1 = np.where(acquired == 1)
    cnt = Counter(axis0)

    def f(x, y):
        if type(x) is not list:
            x = list(range(cnt[x]))
        return x + list(range(cnt[y]))

    def f_mnist_axis0(x, y):
        if type(x) is not list:
            x = [x] * cnt[x] * 49
        return x + [y] * cnt[y] * 49

    def f_mnist_axis1(x, y):
        if type(x) is not list:
            x = [x] * 49
        return x + [y] * 49

    def f_mnist_axis2(x, y):
        if type(x) is not np.ndarray:
            x = mnist_feature_where[x]
        return np.concatenate((x, mnist_feature_where[y]))

    axis1_ = reduce(lambda x, y: f(x, y), np.sort(list(cnt))) # feature index
    # tensorflow mask
    mask[axis0, axis1_] = 1
    mask_ind = np.vstack(np.where(mask)).astype(np.int64).T

    acquired_tmp = np.zeros([data.shape[0], n_features, n_features])
    acquired_tmp[axis0, axis1_, axis1] = acquired[axis0, axis1]

    if data_type=='cube':
        # in the future, we need to handle different number of
        data_tmp = np.zeros([data.shape[0], n_features, n_features])
        data_tmp[axis0, axis1_, axis1] = data[axis0, axis1]
        return np.concatenate((data_tmp, acquired_tmp), axis=2), mask_ind
    elif data_type == 'mnist': # MNIST
        assert n_features == 16
        axis0_mnist = reduce(lambda x, y: f_mnist_axis0(x, y), np.sort(list(cnt)))
        axis1_mnist = reduce(lambda x, y: f_mnist_axis1(x, y), axis1_)
        axis2_mnist = reduce(lambda x, y: f_mnist_axis2(x, y), axis1)
        data_tmp = np.zeros([data.shape[0], n_features, 49])
        assert len(axis0_mnist) == len(axis1_mnist) == len(axis2_mnist)
        data_tmp[axis0_mnist, axis1_mnist, len(axis0) * list(range(49))] = data[axis0_mnist, axis2_mnist]
        return np.concatenate((data_tmp, acquired_tmp), axis=2), mask_ind

def masking_mnist(list):
    msk = np.zeros((28, 28), dtype=np.uint8)
    for idx in list:
        j = int(idx % 4)
        i = int(idx // 4)
        msk[7*i: 7*i + 7, 7*j: 7*j + 7] = np.ones((7, 7), dtype=np.uint8)
    return msk.reshape(-1)

def masking_mnist_from_acquired(list):
    acquired_indices = [i for i, v in enumerate(list) if v==1]
    return masking_mnist(acquired_indices)

def mnist_expand(x, size):
    shape = [28 * size, 28 * size]
    datum = np.zeros(shape)
    ind = np.random.randint(size*size)
    ind_row = ind // size
    ind_col = ind % size
    datum[(ind_row*28):(ind_row*28+28), (ind_col*28):(ind_col*28+28)] = x.reshape(28, 28)
    return datum
