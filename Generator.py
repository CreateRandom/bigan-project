from chainer import Chain
import chainer.functions as F
import chainer.links as L

class Generator(Chain):
    def __init__(self,n_hidden=1024,non_linearity=F.relu):
        super(Generator, self).__init__()

        self.n_hidden = n_hidden
        self.non_linearity = non_linearity
        with self.init_scope():

            self.l0 = L.Linear(None, self.n_hidden)
            self.l1 = L.Linear(self.n_hidden, self.n_hidden)
            self.bn = L.BatchNormalization(self.n_hidden)
            self.l2 = L.Linear(self.n_hidden, 28*28)

    def __call__(self, x):
        l0_out = self.non_linearity(self.l0(x))
        l1_out = self.l1(l0_out)
        bn_out = self.non_linearity(self.bn(l1_out))
        y = F.relu(self.l2(bn_out))
        y = F.reshape(y, (-1, 1, 28, 28))

        return y