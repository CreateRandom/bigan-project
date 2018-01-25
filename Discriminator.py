from chainer import Chain
import chainer.functions as F
import chainer.links as L

class Discriminator(Chain):
    def __init__(self,latent_dim,n_hidden=1024, non_linearity=F.relu,use_encoder=True):
        super(Discriminator, self).__init__()

        self.n_hidden = n_hidden
        self.non_linearity = non_linearity
        self.use_encoder = use_encoder
        with self.init_scope():
            #paper
            # self.l0 = L.Linear(None, self.n_hidden)
            # self.t0 = L.Linear(latent_dim, self.n_hidden)
            # self.l1 = L.Linear(self.n_hidden, self.n_hidden)
            # self.t1 = L.Linear(latent_dim, self.n_hidden)
            # self.bn = L.BatchNormalization(self.n_hidden)
            # self.l2 = L.Linear(self.n_hidden, 1)

            # conv. experiment
            self.l0 = L.Convolution2D(None, 1, ksize=5, stride=2)
            self.t0 = L.Linear(latent_dim, 144)
            self.l1 = L.Linear(None, self.n_hidden)
            self.t1 = L.Linear(latent_dim, self.n_hidden)
            self.l2 = L.Linear(None, 1)


    def __call__(self, input):
        #turn image into flat vector
        # -1 --> filler for arbitrary number of elements, 1 * 28 * 28 --> one color channel, 28 by 28 pixels

      #  x = F.reshape(input[0], (-1, 28*28))
        z = input[1]
        # paper
        # l0_out = self.l0(x)
        #
        # if(self.use_encoder):
        #     t0_out = self.t0(z)
        #     comb_0 = self.non_linearity(l0_out + t0_out)
        # else:
        #     comb_0 = self.non_linearity(l0_out)
        # l1_out = self.l1(comb_0)
        # if(self.use_encoder):
        #     t1_out = self.t1(z)
        #     comb_1 = self.non_linearity(self.bn(l1_out + t1_out))
        # else:
        #     comb_1 = self.non_linearity(self.bn(l1_out))
        # l2_out = self.l2(comb_1)
        # y = F.leaky_relu(l2_out, 0.2)

        #original
        l0_out = self.l0(F.reshape(input[0], (-1, 1, 28, 28)))
        if(self.use_encoder):
            t0_out = self.t0(z)
            comb_1 = self.non_linearity(F.reshape(l0_out,(-1, 144)) + t0_out)
        else:
            comb_1 = self.non_linearity(F.reshape(l0_out,(-1, 144)))
        l1_out = self.l1(comb_1)
        if(self.use_encoder):
            t1_out = self.t1(z)
            comb_2 = self.non_linearity(l1_out + t1_out)
        else:
            comb_2 = self.non_linearity(l1_out)
        y = self.l2(comb_2)

        return y