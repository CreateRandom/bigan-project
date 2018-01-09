from chainer import Chain
import chainer.functions as F
import chainer.links as L

class Generator(Chain):
    def __init__(self):
        super(Generator, self).__init__()

        self.in_channels = 20

        with self.init_scope():
            self.l0 = L.Linear(None, self.in_channels*784)
            self.l1 = L.BatchNormalization(self.in_channels*784)
            self.l2 = L.Deconvolution2D(self.in_channels, 1, ksize=3, outsize=(28,28), stride=1, pad=1)

    def __call__(self, x):
        # fully connected relu layer
        h0 = self.l0(x)
        # batch normalization
        h1 = F.relu(self.l1(h0))
        # reshape for deconv layer
        h1 = F.reshape(h1,(-1, self.in_channels, 28,28))
        # sigmoid deconv
        h2 = self.l2(h1)
        y = F.sigmoid(h2)
        return y