from chainer import Chain
import chainer.functions as F
import chainer.links as L

class Generator(Chain):
    def __init__(self,n_hidden=1024,non_linearity=F.relu):
        super(Generator, self).__init__()

        self.n_hidden = n_hidden
        self.non_linearity = non_linearity
        self.in_channels = 1
        self.img_size = 28
        with self.init_scope():
            # paper
            # self.l0 = L.Linear(None, self.n_hidden)
            # self.l1 = L.Linear(self.n_hidden, self.n_hidden)
            # self.bn = L.BatchNormalization(self.n_hidden)
            # self.l2 = L.Linear(self.n_hidden, 28*28)

            # # conv. exper
            self.l0 = L.Linear(None, self.n_hidden * 4 * 4)
            self.l1 = L.Deconvolution2D(self.n_hidden, self.n_hidden/2, ksize=7, outsize=(8,8), stride=1, pad=1)
            self.l2 = L.Deconvolution2D(self.n_hidden/2, self.n_hidden/4, ksize=5, outsize=(20,20), stride=2, pad=0)
            self.l3 = L.Deconvolution2D(self.n_hidden/4, 1, ksize=9, outsize=(self.img_size, self.img_size), stride=1, pad=0)

            # self.l0 = L.Linear(None, self.n_hidden)
            # self.l1 = L.Linear(None, self.n_hidden)
            # self.l2 = L.Linear(None, self.img_size*self.img_size)


    def __call__(self, x):
        # paper
        # l0_out = self.non_linearity(self.l0(x))
        # l1_out = self.l1(l0_out)
        # bn_out = self.non_linearity(self.bn(l1_out))
        # y = F.relu(self.l2(bn_out))

        # conv. exper
        h1 = F.reshape(self.l0(x),(-1, self.n_hidden, 4, 4))
        h2 = self.l1(h1)
        #h2 = self.non_linearity(h2)
        h3 = self.l2(h2)
        #h3 = self.non_linearity(h3)
        y = self.l3(h3)
        #y = self.non_linearity(y)

        #y = F.reshape(h3, (-1, 1, 28, 28))

        return y