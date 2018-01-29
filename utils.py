import numpy as np
from chainer.dataset import download
import os
import gzip
import numpy
import struct
import six
import chainer
from chainer.datasets import TupleDataset
import chainer.datasets.mnist as mnist

def rescale_noise(x):
    x = (x * 2)
    x = x - 1
    return x

def shuffle_in_unison(a, b, c):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(c)

def get_emnist(withlabel=True, ndim=1, scale=1., dtype=numpy.float32,
              label_dtype=numpy.int32, rgb_format=False):
    train_raw = _retrieve_emnist_training()
    train = mnist._preprocess_mnist(train_raw, withlabel, ndim, scale, dtype,
                              label_dtype, rgb_format)
    test_raw = _retrieve_emnist_test()
    test = mnist._preprocess_mnist(test_raw, withlabel, ndim, scale, dtype,
                             label_dtype, rgb_format)
    return train, test

def _retrieve_emnist_training():
    archives = ['bin/emnist-letters-train-images-idx3-ubyte.gz',
            'bin/emnist-letters-train-labels-idx1-ubyte.gz']
    return _retrieve_emnist('em_train.npz', archives)

def _retrieve_emnist_test():
    archives = ['bin/emnist-letters-test-images-idx3-ubyte.gz',
            'bin/emnist-letters-test-labels-idx1-ubyte.gz']
    return _retrieve_emnist('em_test.npz', archives)

def _retrieve_emnist(name, archives):
    # the path to store the cached file to
    root = download.get_dataset_directory('pfnet/chainer/emnist')
    path = os.path.join(root, name)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, archives), numpy.load)

def _make_npz(path,archives):
    x_url, y_url = archives

    with gzip.open(x_url, 'rb') as fx, gzip.open(y_url, 'rb') as fy:
        fx.read(4)
        fy.read(4)
        N, = struct.unpack('>i', fx.read(4))
        if N != struct.unpack('>i', fy.read(4))[0]:
            raise RuntimeError('wrong pair of EMNIST images and labels')
        fx.read(8)

        x = numpy.empty((N, 784), dtype=numpy.uint8)
        y = numpy.empty(N, dtype=numpy.uint8)

        for i in six.moves.range(N):
            y[i] = ord(fy.read(1))
            for j in six.moves.range(784):
                x[i, j] = ord(fx.read(1))

    numpy.savez_compressed(path, x=x, y=y)
    return {'x': x, 'y': y}

def get_mnist(n_train=100, n_test=100, n_dim=1, with_label=True, classes = None):
    """

    :param n_train: nr of training examples per class
    :param n_test: nr of test examples per class
    :param n_dim: 1 or 3 (for convolutional input)
    :param with_label: whether or not to also provide labels
    :param classes: if not None, then it selects only those classes, e.g. [0, 1]
    :return:
    """

    train_data, test_data = chainer.datasets.get_mnist(ndim=n_dim, withlabel=with_label)

    if not classes:
        classes = np.arange(10)
    n_classes = len(classes)

    if with_label:

        for d in range(2):

            if d==0:
                data = train_data._datasets[0]
                labels = train_data._datasets[1]
                n = n_train
            else:
                data = test_data._datasets[0]
                labels = test_data._datasets[1]
                n = n_test

            for i in range(n_classes):
                lidx = np.where(labels == classes[i])[0][:n]
                if i==0:
                    idx = lidx
                else:
                    idx = np.hstack([idx,lidx])

            L = np.concatenate([i*np.ones(n) for i in np.arange(n_classes)]).astype('int32')

            if d==0:
                train_data = TupleDataset(data[idx],L)
            else:
                test_data = TupleDataset(data[idx],L)

    else:

        tmp1, tmp2 = chainer.datasets.get_mnist(ndim=n_dim,withlabel=True)

        for d in range(2):

            if d == 0:
                data = train_data
                labels = tmp1._datasets[1]
                n = n_train
            else:
                data = test_data
                labels = tmp2._datasets[1]
                n = n_test

            for i in range(n_classes):
                lidx = np.where(labels == classes[i])[0][:n]
                if i == 0:
                    idx = lidx
                else:
                    idx = np.hstack([idx, lidx])

            if d == 0:
                train_data = data[idx]
            else:
                test_data = data[idx]

    return train_data, test_data



# Custom iterator
class RandomIterator(object):
    """
    Generates random subsets of data
    """

    def __init__(self, data, batch_size=1):
        """

        Args:
            data (TupleDataset):
            batch_size (int):

        Returns:
            list of batches consisting of (input, output) pairs
        """

        self.data = data

        self.idx = 0

        self.batch_size = batch_size
        self.n_batches = len(self.data) // batch_size

    def __iter__(self):

        self.idx = -1
        self._order = np.random.permutation(len(self.data))[:(self.n_batches * self.batch_size)]

        return self

    def next(self):

        self.idx += 1

        if self.idx == self.n_batches:
            raise StopIteration

        i = self.idx * self.batch_size

        # handles unlabeled and labeled data
        if isinstance(self.data, np.ndarray):
            return self.data[self._order[i:(i + self.batch_size)]]
        else:
            return list(self.data[self._order[i:(i + self.batch_size)]])


