import utils as u
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from chainer import optimizers
from chainer import functions as f
from chainer import Variable
from chainer import iterators as i
from CustomClassifier import CustomClassifier
import Discriminator as d
import Generator as g
import Encoder as e


n_train = 1000
batchsize = 10
train_data, test_data = u.get_mnist(n_train=n_train, n_test=100, with_label=False, classes = [0])
#train_noise = np.random.rand(n_train, 25).astype(np.float32)

# the discriminator is tasked with classifying examples as real or fake
Disc = CustomClassifier(predictor=d.Discriminator(), lossfun=f.sigmoid_cross_entropy)
Disc.compute_accuracy = False
Gen = g.Generator()
Enc = e.Encoder()

# Use Adam optimizer
disc_optimizer = optimizers.Adam(0.001)
gen_optimizer = optimizers.Adam(0.001)
enc_optimizer = optimizers.Adam(0.001)
# Optimize the loss
disc_optimizer.setup(Disc)
gen_optimizer.setup(Gen)
enc_optimizer.setup(Enc)

#Define iterator
train_iter_X = i.SerialIterator(train_data,batch_size=batchsize,repeat=True,shuffle=True)

n_epoch = 10
disc_loss_list = []
gen_loss_list = []
enc_loss_list = []

for i in xrange(0, n_epoch):
    print i
    n_batches = n_train / batchsize
    for k in xrange(0, n_batches):
        loss_on_real = 0
        loss_on_fake = 0
        gen_loss_item = 0
        enc_loss_item = 0
        # create z
        train_noise = np.random.rand(batchsize, 25).astype(np.float32)
        # create G(z)
        fakeImages = Gen(Variable(train_noise))
        # create x
        realImages = train_iter_X.next()
        # create E(x)
        projectedImages = Enc(Variable(np.array(realImages)))

        # Update the discriminator
        #disc_optimizer.update(Disc, fakeImages, np.zeros((len(fakeImages), 1)).astype(np.int32))
        #disc_optimizer.update(Disc, Variable(np.array(realImages)), np.ones((len(realImages), 1)).astype(np.int32))


        disc_optimizer.update(Disc, (fakeImages, train_noise), np.zeros((len(fakeImages), 1)).astype(np.int32))
        disc_optimizer.update(Disc, (Variable(np.array(realImages)),projectedImages), np.ones((len(realImages), 1)).astype(np.int32))




        # Keep track of loss for plotting
        loss_on_real = loss_on_real + Disc((fakeImages, train_noise), np.ones((len(realImages), 1)).astype(np.int32)).data
        loss_on_fake = loss_on_fake + Disc((Variable(np.array(realImages)),projectedImages), np.ones((len(realImages), 1)).astype(np.int32)).data

        # Get predictions on fake images
        predictions = Disc.get_predictions((fakeImages, train_noise))

        # Update generator
        gen_loss = f.sigmoid_cross_entropy(predictions, np.ones((len(realImages), 1)).astype(np.int32))
        gen_loss_item = gen_loss_item + 2 * gen_loss.data
        Gen.cleargrads()
        gen_loss.backward()
        gen_loss.unchain_backward()
        gen_optimizer.update()

        # get predictions on real images
        predictions = Disc.get_predictions((Variable(np.array(realImages)), projectedImages))

        # Update Encoder
        enc_loss = f.sigmoid_cross_entropy(predictions, np.zeros((len(realImages), 1)).astype(np.int32))
        enc_loss_item = enc_loss_item + 2 * enc_loss.data
        Enc.cleargrads()
        enc_loss.backward()
        enc_loss.unchain_backward()
        enc_optimizer.update()

        # Add loss values to the lists for plotting
        disc_loss_list.append(loss_on_real + loss_on_fake)
        gen_loss_list.append(gen_loss_item)
        enc_loss_list.append(enc_loss_item)

#Plot loss over epochs
plt.plot(np.arange(1,n_epoch*n_batches+1),disc_loss_list,'b-',np.arange(1,n_epoch*n_batches+1),gen_loss_list,'g-')
plt.title("Generator and Discriminator loss")
plt.ylim(0,2.0)
plt.xlim(1,n_epoch*n_batches+1)
plt.xlabel("Batch")
plt.ylabel("Loss")
blue = mpatches.Patch(color='blue', label='Discriminator')
green = mpatches.Patch(color='green', label='Generator')
plt.legend(handles=[blue,green])
plt.show()

# create five generated images with the generato
# sample some noise
noise = np.random.rand(10, 25).astype(np.float32)
# generate fake samples using the generator
generatedImages = Gen(Variable(noise))

#Show generated images
f, subplot = plt.subplots(2, 5)
for i in range(0, 5):
    subplot[0, i].imshow((generatedImages._data[0][i].astype(np.float64).reshape(28, 28)), cmap='gray')
    subplot[1, i].imshow((train_data[i].reshape(28, 28)), cmap='gray')
plt.show()