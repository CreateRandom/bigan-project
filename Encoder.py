from chainer import Chain
import chainer.functions as F
import chainer.links as L

class Encoder(Chain):
    def __init__(self,latent_dim,n_hidden=1024):
        super(Encoder, self).__init__()

        self.n_hidden = n_hidden

        with self.init_scope():
            self.l0 = L.Linear(None, self.n_hidden)
            self.l1 = L.Linear(self.n_hidden, self.n_hidden)
            self.bn = L.BatchNormalization(self.n_hidden)
            self.l2 = L.Linear(self.n_hidden, latent_dim)

    def __call__(self, x):
        # -1 --> filler for arbitrary number of elements, 1 * 28 * 28 --> one color channel, 28 by 28 pixels
        x = F.reshape(x, (-1, 28*28))
        l0_out = F.relu(self.l0(x))
        l1_out = self.l1(l0_out)
        bn_out = F.relu(self.bn(l1_out))
        y = F.relu(self.l2(bn_out))

        return y