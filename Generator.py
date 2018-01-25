from chainer import Chain
import chainer.functions as F
import chainer.links as L

class Generator(Chain):
    def __init__(self,n_hidden=1024,non_linearity=F.relu):
        super(Generator, self).__init__()

        self.n_hidden = n_hidden
        self.non_linearity = non_linearity
        self.in_channels  = 5
        self.img_size = 28
        with self.init_scope():
            # paper
            # self.l0 = L.Linear(None, self.n_hidden)
            # self.l1 = L.Linear(self.n_hidden, self.n_hidden)
            # self.bn = L.BatchNormalization(self.n_hidden)
            # self.l2 = L.Linear(self.n_hidden, 28*28)

            # conv. exper
            self.l0 = L.Linear(None, self.in_channels*self.img_size*self.img_size)
            self.l1 = L.BatchNormalization(self.in_channels*self.img_size*self.img_size)
            self.l2 = L.Deconvolution2D(self.in_channels, 1, ksize=3, outsize=(self.img_size,self.img_size), stride=1, pad=1)

    def __call__(self, x):
        # paper
        # l0_out = self.non_linearity(self.l0(x))
        # l1_out = self.l1(l0_out)
        # bn_out = self.non_linearity(self.bn(l1_out))
        # y = F.relu(self.l2(bn_out))

        # conv. exper
        # fully connected relu layer
        h0 = self.l0(x)
        h1 = self.non_linearity(self.l1(h0))
        # reshape for deconv layer
        h1 = F.reshape(h1,(-1, self.in_channels, self.img_size,self.img_size))
        # sigmoid deconv
        h2 = self.l2(h1)
        y = F.sigmoid(h2)

        y = F.reshape(y, (-1, 1, 28, 28))

        return y