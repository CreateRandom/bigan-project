import utils as u
import matplotlib.pyplot as plt
import numpy as np
from chainer import optimizers
from chainer import links as l
from chainer import functions as f

from chainer import Variable
from chainer import iterators as i
from CustomClassifier import CustomClassifier
import Discriminator as d
import Generator as g


n_train = 1000
batchsize = 40
train_data, test_data = u.get_mnist(n_train=n_train, n_test=100, with_label=False, classes = [0])
train_noise = np.random.rand(n_train, 784).astype(np.float32)
train_noise2 = np.random.rand(n_train, 784).astype(np.float32)

test_noise = np.random.rand(100, 28*28).astype(np.float32)

# the discriminator is tasked with classifying examples as real or fake
Disc = CustomClassifier(predictor=d.Discriminator(), lossfun=f.sigmoid_cross_entropy)
Disc.compute_accuracy = False

Gen = g.Generator()

#plt.imshow((train_data[99].reshape(28, 28)), cmap='gray')
# plt.show()

# Use stochastic gradient descent
disc_optimizer = optimizers.Adam()
gen_optimizer = optimizers.Adam()
# Optimize the loss
disc_optimizer.setup(Disc)
gen_optimizer.setup(Gen)

#train_iter_X = u.RandomIterator(train_data, batch_size=32)
train_iter_X = i.SerialIterator(train_data,batch_size=batchsize,repeat=True,shuffle=True)
test_iter_X = u.RandomIterator(test_data, batch_size=32)

tester = i.SerialIterator(train_noise,batch_size=batchsize,repeat=True,shuffle=True)
tester2 = i.SerialIterator(train_noise2,batch_size=batchsize,repeat=True,shuffle=True)

#train_iter_Z = u.RandomIterator(train_noise, batch_size=32)
#test_iter_Z = u.RandomIterator(test_noise, batch_size=32)

n_epoch = 10
disc_loss = []
gen_loss_list = []

for i in xrange(0, n_epoch):
    # iterate over batches
    n_batches = n_train / batchsize
    loss_on_real = 0
    loss_on_fake = 0
    gen_loss_item = 0
    for k in xrange(0,n_batches):
        # generate fake samples using the generator
        rand = np.array(tester.next())
        fakeImages = Gen(Variable(rand))
        # obtain real images
        realImages = train_iter_X.next()

        # # update the discriminator with the current fake images
        # disc_loss_fake = f.sigmoid_cross_entropy(fakeImages,np.ones((len(fakeImages), 1)).astype(np.int32)) #f.sum( f.log(1 - Disc.get_predictions(fakeImages)))
        # disc_loss_real = f.sigmoid_cross_entropy(Variable(np.array(realImages)), np.zeros((len(realImages), 1)).astype(np.int32)) #f.sum( f.log(Disc.get_predictions(Variable(np.array(realImages)))))
        #
        # Disc.cleargrads()
        # disc_loss_fake.backward()
        # disc_loss_fake.unchain_backward()
        # disc_optimizer.update()
        #
        # Disc.cleargrads()
        # disc_loss_real.backward()
        # disc_loss_real.unchain_backward()
        # disc_optimizer.update()

        disc_optimizer.update(Disc, fakeImages, np.ones((len(fakeImages), 1)).astype(np.int32))
        disc_optimizer.update(Disc, Variable(np.array(realImages)), np.zeros((len(realImages), 1)).astype(np.int32))

        loss_on_real = loss_on_real + Disc(Variable(np.array(realImages)), np.ones((len(realImages), 1)).astype(np.int32)).data / n_batches
        loss_on_fake = loss_on_fake + Disc(fakeImages, np.zeros((len(fakeImages), 1)).astype(np.int32)).data / n_batches

        # get some more noise
        fakeImages2 = Gen(Variable(np.array(tester2.next())))
        predictions = Disc.get_predictions(fakeImages2)
        gen_loss = f.sum(f.log(1 - predictions))
        gen_loss_item = gen_loss_item + gen_loss.data / n_batches
        # do backprop
        Gen.cleargrads()
        gen_loss.backward()
        gen_loss.unchain_backward()
        gen_optimizer.update()
    disc_loss.append(loss_on_real + loss_on_fake)
    gen_loss_list.append(gen_loss_item)

plt.plot(np.arange(1,n_epoch+1),disc_loss,'b-',np.arange(1,n_epoch+1),gen_loss_list,'g-')
plt.show()

fakeImages = Gen(Variable(train_noise2))
test = fakeImages._data[0][0]
plt.imshow((test.astype(np.float64).reshape(28, 28)), cmap='gray')
plt.show()
