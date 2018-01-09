from chainer import Chain
import chainer.functions as F
import chainer.links as L

class Discriminator(Chain):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_hidden = 1500

        with self.init_scope():
            self.l0 = L.Convolution2D(None, 1, ksize=5, stride=2)
            self.t0 = L.Linear(None, 144)
            self.l1 = L.Linear(None, self.n_hidden)
            self.t1 = L.Linear(None, self.n_hidden)
            self.l2 = L.Linear(None, 1)

    def __call__(self, input):
        x = input[0]
        z = input[1]
        # -1 --> filler for arbitrary number of elements, 1 * 28 * 28 --> one color channel, 28 by 28 pixels
        l0_out = self.l0(F.reshape(x, (-1, 1, 28, 28)))
        t0_out = self.t0(z)
        comb_1 = F.relu(F.reshape(l0_out,(-1, 144)) + t0_out)
        l1_out = self.l1(comb_1)
        t1_out = self.t1(z)
        comb_2 = F.relu(l1_out + t1_out)
        y = self.l2(comb_2)
        return y