import numpy as np
from chainer import Chain
import chainer.functions as F
import chainer.links as L


class Generator(Chain):
    def __init__(self):
        super(Generator, self).__init__()

        with self.init_scope():
            self.l0 = L.Linear(None, 28*28)
            self.l05 = L.BatchNormalization(784)
            self.l1 = L.Deconvolution2D(1, 1)

    def __call__(self, x):
        # fully connected relu layer
        h1 = F.relu(self.l0(x))
        # batch normalization
        h1 = self.l05(h1)
       # h1 = F.batch_normalization(h1, np.ones(h1.shape[1:-1]).astype(np.float32),np.ones(h1.shape[1:-1]).astype(np.float32))
        # reshape for deconv layer
        h1 = F.reshape(h1,(-1, 1, 28,28))
        h2 = self.l1(h1)
        y = F.sigmoid(h2)
        return y