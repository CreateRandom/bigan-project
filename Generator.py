from chainer import Chain
import chainer.functions as F
import chainer.links as L

class Generator(Chain):
    def __init__(self):
        super(Generator, self).__init__()

        self.img_size = 28
        self.in_channels = 30

        with self.init_scope():
            self.l0 = L.Linear(None, self.in_channels*self.img_size*self.img_size)
            self.l1 = L.BatchNormalization(self.in_channels*self.img_size*self.img_size)
            self.l2 = L.Deconvolution2D(self.in_channels, 1, ksize=3, outsize=(self.img_size,self.img_size), stride=1, pad=1)

    def __call__(self, x):
        # fully connected relu layer
        h0 = self.l0(x)
        # batch normalization
        h1 = F.relu(self.l1(h0))
        # reshape for deconv layer
        h1 = F.reshape(h1,(-1, self.in_channels, self.img_size,self.img_size))
        # sigmoid deconv
        h2 = self.l2(h1)
        y = F.sigmoid(h2)
        return y