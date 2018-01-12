from chainer import Chain
import chainer.functions as F
import chainer.links as L

class Encoder(Chain):
    def __init__(self):
        super(Encoder, self).__init__()
        self.n_hidden = 1000
        with self.init_scope():
            self.l0 = L.Convolution2D(None, 1, ksize=5, stride=2)
            self.l1 = L.Linear(None, self.n_hidden)
            self.l2 = L.Linear(None, self.n_hidden)
            self.l3 = L.Linear(None, 25)

    def __call__(self, x):
        # -1 --> filler for arbitrary number of elements, 1 * 28 * 28 --> one color channel, 28 by 28 pixels
        re = F.relu(self.l0(F.reshape(x, (-1, 1, 28, 28))))
        h1 = F.relu(self.l2(re))
        y = self.l3(h1)
        return y