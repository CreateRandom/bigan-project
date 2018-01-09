from chainer import Chain
import chainer.functions as F
import chainer.links as L


class Discriminator(Chain):
    # override init
    def __init__(self):
        super(Discriminator, self).__init__()

        with self.init_scope():
            self.l0 = L.Convolution2D(None, 1, ksize=5, stride=2)
            self.l1 = L.Linear(None, 1)

    def __call__(self, x):
        # -1 --> filler for arbitrary number of elements, 1 * 28 * 28 --> one color channel, 28 by 28 pixels
        re = F.reshape(x, (-1, 1, 28, 28))
        h1 = F.relu(self.l0(re))
        y = self.l1(h1)
        return y