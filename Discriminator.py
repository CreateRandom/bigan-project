from chainer import Chain
import chainer.functions as F
import chainer.links as L

class Discriminator(Chain):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_hidden = 1000
        self.n_dim = 28

        with self.init_scope():

            #original
            self.l0 = L.Convolution2D(None, 1, ksize=5, stride=2)
            self.t0 = L.Linear(None, 144)
            self.l1 = L.Linear(None, self.n_hidden)
            self.t1 = L.Linear(None, self.n_hidden)
            self.l2 = L.Linear(None, 1)

            #test 1

            #self.l0 = L.Convolution2D(None, 1, ksize=8, stride=2)

            #self.lz0 = L.Linear(None, self.n_dim*121)
            #self.lz1 = L.Deconvolution2D(self.n_dim, 1, ksize=1, stride=1, pad=0)

            #self.l1 = L.Linear(None, self.n_hidden)
            #self.classify = L.Linear(None, 1)

            # test 2

            #self.l0 = L.Convolution2D(None, 1, ksize=10, stride=4)
            #self.l1 = L.Linear(None, 25)
            #self.l2 = L.Linear(None, 25)
            #self.l3 = L.Linear(None, 1)

    def __call__(self, input):
        x = input[0]
        z = input[1]
        # -1 --> filler for arbitrary number of elements, 1 * 28 * 28 --> one color channel, 28 by 28 pixels

        #original
        l0_out = self.l0(F.reshape(x, (-1, 1, 28, 28)))
        t0_out = self.t0(z)
        comb_1 = F.relu(F.reshape(l0_out,(-1, 144)) + t0_out)
        l1_out = self.l1(comb_1)
        t1_out = self.t1(z)
        comb_2 = F.relu(l1_out + t1_out)
        y = self.l2(comb_2)

        #test 1

        #convolution on the image
        #i = self.l0(x.reshape(128,1,28,28))
        #i = self.l1(i_0)

        #deconv on latent space rep
        #n0 = F.reshape(self.lz0(z),(-1, self.n_dim, 11,11))
        #n = self.lz1(n0)

        #add the image and latent space rep (5x5 object now)
        #comb = F.relu(i + n)
        #comb_to_1000 = self.l1(comb)

        #classify
        #y = self.classify(comb)

        #test 2
        #x = self.l0(F.reshape(x, (-1, 1, 28, 28)))
        #comb_1 = x + F.reshape(z,(-1, 1, 5, 5))
        #l1_out = self.l1(comb_1)
        #comb_2 = l1_out + z
        #l2_out = self.l2(comb_2)
        #comb_3 = l2_out + z
        #y = self.l3(comb_3)



        return y