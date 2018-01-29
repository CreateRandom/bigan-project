import chainer

import utils as u
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from chainer import serializers
from chainer import optimizers
from chainer import optimizer
from chainer import functions as f
from chainer import Variable
from chainer import iterators as i
from chainer import datasets as chsets
from chainer import using_config
from chainer import training as training
from CustomClassifier import CustomClassifier
import Discriminator as d
import Generator as g
import Encoder as e
import math
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
n_epoch = 20
n_train_per_class = 512
batchsize = 128
weight_decay = 0.000025
initial_alpha = 0.0002
final_alpha = 0.000002
beta_1 = 0.5
beta_2 = 0.999
latent_dim = 50
train_data, test_data = u.get_mnist(n_train=n_train_per_class, n_test=512, with_label=True, classes = [0,1,2,3,4,5,6,7,8,9])

#train_data, test_data = chsets.get_mnist()
#train_data, test_data = u.get_emnist()
# get the correct total length
n_train = train_data._length

# the discriminator is tasked with classifying examples as real or fake
Disc = CustomClassifier(predictor=d.Discriminator(latent_dim,n_hidden=8,non_linearity=f.relu,
                                                  use_encoder=True), lossfun=f.sigmoid_cross_entropy)
Disc.compute_accuracy = False
Gen = g.Generator(n_hidden=8,non_linearity=f.relu)
Enc = e.Encoder(latent_dim,n_hidden=8,non_linearity=f.relu)

# Use Adam optimizer
# learning rate, beta1, beta2
disc_optimizer = optimizers.Adam(initial_alpha,beta1=beta_1,beta2=beta_2)
disc_optimizer.setup(Disc)

gen_optimizer = optimizers.Adam(initial_alpha,beta1=beta_1,beta2=beta_2)
gen_optimizer.setup(Gen)

enc_optimizer = optimizers.Adam(initial_alpha,beta1=beta_1,beta2=beta_2)
enc_optimizer.setup(Enc)

# use weight decay if specified
if weight_decay is not None:
    # using those parameters for all optimizers: 0.36 --> 0.61
    disc_optimizer.add_hook(optimizer.WeightDecay(rate=weight_decay))
    gen_optimizer.add_hook(optimizer.WeightDecay(rate=weight_decay))
    enc_optimizer.add_hook(optimizer.WeightDecay(rate=weight_decay))

#Define iterator
train_iter_X = i.SerialIterator(train_data,batch_size=batchsize,repeat=True,shuffle=True)

# exponential learning rate decay
# decrease the rate after half of the epochs have passed by such a rate that it reaches the final alpha at the end
factor = math.pow((final_alpha / initial_alpha),  (1 / math.floor(n_epoch * 0.5)))
shift1 = training.extensions.ExponentialShift(attr="alpha",rate=factor,init=initial_alpha,target=final_alpha,optimizer=disc_optimizer)
shift1.initialize(None)
shift2 = training.extensions.ExponentialShift(attr="alpha",rate=factor,init=initial_alpha,target=final_alpha,optimizer=gen_optimizer)
shift2.initialize(None)
shift3 = training.extensions.ExponentialShift(attr="alpha",rate=factor,init=initial_alpha,target=final_alpha,optimizer=enc_optimizer)
shift3.initialize(None)

disc_loss_list = []
disc_loss_list2 = []

gen_loss_list = []
enc_loss_list = []

for i in xrange(0, n_epoch):
    # learning rate decay
    if i > math.floor(n_epoch * 0.5):
        shift1(None)
        shift2(None)
        shift3(None)
    print i
    n_batches = n_train / batchsize
    for k in xrange(0, n_batches):
        loss_on_real = 0
        loss_on_fake = 0
        gen_loss_item = 0
        enc_loss_item = 0
        # create z
        train_noise = np.random.rand(batchsize, latent_dim).astype(np.float32)
        # create G(z)
        fakeImages = Gen(Variable(train_noise))
        # create x
        realImages = train_iter_X.next()
        latent = []
        for image in realImages:
            latent.append(image[0])
        realImages = latent
        # create E(x)
        projectedImages = Enc(Variable(np.array(realImages)))

        # Update the discriminator
        #disc_optimizer.update(Disc, fakeImages, np.zeros((len(fakeImages), 1)).astype(np.int32))
        #disc_optimizer.update(Disc, Variable(np.array(realImages)), np.ones((len(realImages), 1)).astype(np.int32))


        disc_optimizer.update(Disc, (fakeImages, train_noise), np.zeros((len(fakeImages), 1)).astype(np.int32))
        disc_optimizer.update(Disc, (Variable(np.array(realImages)),projectedImages), np.ones((len(realImages), 1)).astype(np.int32))


        # Keep track of loss for plotting
        loss_on_real = loss_on_real + Disc((fakeImages, train_noise), np.zeros((len(realImages), 1)).astype(np.int32)).data
        loss_on_fake = loss_on_fake + Disc((Variable(np.array(realImages)),projectedImages), np.ones((len(realImages), 1)).astype(np.int32)).data

        # Get predictions on fake images
        predictions = Disc.get_predictions((fakeImages, train_noise))

        # Update generator
        gen_loss = f.sigmoid_cross_entropy(predictions, np.ones((len(fakeImages), 1)).astype(np.int32))
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
        disc_loss_list.append(loss_on_real)
        disc_loss_list2.append(loss_on_fake)
        gen_loss_list.append(gen_loss_item)
        enc_loss_list.append(enc_loss_item)

#Plot loss over epochs
plt.plot(np.arange(1,n_epoch*n_batches+1),disc_loss_list,'b-',np.arange(1,n_epoch*n_batches+1),gen_loss_list,'g-',
    np.arange(1, n_epoch * n_batches + 1), enc_loss_list, 'r-',np.arange(1,n_epoch*n_batches+1),disc_loss_list2,'y-')
plt.title("Generator and Discriminator loss")
plt.ylim(0,3)
plt.xlim(1,n_epoch*n_batches+1)
plt.xlabel("Batch")
plt.ylabel("Loss")
blue = mpatches.Patch(color='blue', label='Disc (real)')
green = mpatches.Patch(color='green', label='Generator')
red = mpatches.Patch(color='red', label='Encoder')
yellow = mpatches.Patch(color='yellow', label='Disc (fake)')
plt.legend(handles=[blue,green,red,yellow])
plt.show()

save = True
if(save):
    serializers.save_npz('disc',Disc)
    serializers.save_npz('gen',Gen)
    serializers.save_npz('enc',Enc)

# end of training --> set context to make BN layers behave properly
#using_config('train',False)
# use this call instead as the other seems to have no effect for unknown reasons...
chainer.config.__setattr__('train',False)
# use this to inspect the current config.
#chainer.config.show()


latent = []
imagesOnly = []
labels = []
for image in test_data:
    pixels = image[0]
    latent_img = Enc(pixels)
    latent.append(latent_img._data[0][0])
    imagesOnly.append(pixels)
    labels.append(image[1])


# split test set again
X_train, X_test, y_train, y_test = train_test_split(latent, labels, test_size=0.2, random_state=14)

# use k-nearest neighbours
clf = neighbors.KNeighborsClassifier(1,'uniform')
clf.fit(X_train,y_train)

pred = clf.predict(X_test)

print accuracy_score(pred,y_test)

# create five generated images with the generator
# sample some noise
noise = np.random.rand(5, latent_dim).astype(np.float32)
# generate fake samples using the generator
generatedImages = Gen(Variable(noise))
# map the test data into latent space
latentAll = Enc(Variable(np.array(imagesOnly)))
# build a reconstruction
reconstructed = Gen(latentAll)

#Show generated images
f, subplot = plt.subplots(3, 5)
for i in range(0, 5):
    subplot[0, i].imshow((generatedImages._data[0][i].astype(np.float64).reshape(28, 28)), cmap='gray')
    subplot[1, i].imshow((imagesOnly[i].reshape(28, 28)), cmap='gray')
    subplot[2, i].imshow((reconstructed._data[0][i].astype(np.float64).reshape(28, 28)), cmap='gray')
plt.show()

