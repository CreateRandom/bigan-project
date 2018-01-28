from chainer import Chain
import chainer.functions as F
import chainer.links as L

class Encoder(Chain):
    def __init__(self,latent_dim,n_hidden=1024,non_linearity=F.relu):
        super(Encoder, self).__init__()

        self.n_hidden = n_hidden
        self.non_linearity = non_linearity

        with self.init_scope():
            # paper

            # self.l0 = L.Linear(None, self.n_hidden)
            # self.l1 = L.Linear(self.n_hidden, self.n_hidden)
            # self.bn = L.BatchNormalization(self.n_hidden)
            # self.l2 = L.Linear(self.n_hidden, latent_dim)

            # # conv. experiment
            self.l0 = L.Convolution2D(None, 1, ksize=5, stride=1)
            self.l1 = L.Convolution2D(None, 1, ksize=6, stride=2)
            self.l2 = L.Convolution2D(None, 1, ksize=3, stride=1)
            self.l3 = L.Linear(None, latent_dim)

            # self.l0 = L.Linear(None, self.n_hidden)
            # self.l1 = L.Linear(None, self.n_hidden)
            # self.l2 = L.Linear(None, latent_dim)

    def __call__(self, x):
        # -1 --> filler for arbitrary number of elements, 1 * 28 * 28 --> one color channel, 28 by 28 pixels

        # paper
        # x = F.reshape(x, (-1, 28*28))
        # l0_out = self.non_linearity(self.l0(x))
        # l1_out = self.l1(l0_out)
        # bn_out = self.non_linearity(self.bn(l1_out))
        # l2_out = self.l2(bn_out)
        # y = F.leaky_relu(l2_out, 0.2)

        # conv. experiment

        h0 = F.reshape(x, (-1, 1, 28, 28))
        h1 = self.l0(h0)
        #h1 = self.non_linearity(h1)
        h2 = self.l1(h1)
        #h2 = self.non_linearity(h2)
        h3 = self.l2(h2)
        #h3 = self.non_linearity(h3)
        y = self.l3(h3)
        #y = self.non_linearity(y)

        return y