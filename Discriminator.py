from chainer import Chain
import chainer.functions as F
import chainer.links as L

class Discriminator(Chain):
    def __init__(self,latent_dim,n_hidden=1024):
        super(Discriminator, self).__init__()

        self.n_hidden = n_hidden

        with self.init_scope():
            #paper
            self.l0 = L.Linear(None, self.n_hidden)
            self.t0 = L.Linear(latent_dim, self.n_hidden)
            self.l1 = L.Linear(self.n_hidden, self.n_hidden)
            self.t1 = L.Linear(latent_dim, self.n_hidden)
            self.bn = L.BatchNormalization(self.n_hidden)
            self.l2 = L.Linear(self.n_hidden, 1)




    def __call__(self, input):
        #turn image into flat vector
        # -1 --> filler for arbitrary number of elements, 1 * 28 * 28 --> one color channel, 28 by 28 pixels

        x = F.reshape(input[0], (-1, 28*28))
        z = input[1]
        # paper
        l0_out = self.l0(x)
        t0_out = self.t0(z)
        comb_0 = F.relu(l0_out + t0_out)
        l1_out = self.l1(comb_0)
        t1_out = self.t1(z)
        comb_1 = F.relu(self.bn(l1_out + t1_out))
        l2_out = self.l2(comb_1)
        y = F.leaky_relu(l2_out, 0.2)

        return y