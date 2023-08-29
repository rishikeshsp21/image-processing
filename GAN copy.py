import tensorflow as tf
import cv2
from keras.datasets import mnist
from keras.layers import *
from keras.layers import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt
import keras
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
xtrain = (xtrain - 127.5)/127.5
totalepochs = 50
batchsize = 256
halfbatch = 128
numofbatch = int(xtrain.shape[0]/batchsize)
noisedim = 100
adam = Adam(learning_rate = 2e-4, beta_1 = 0.5)
generator = Sequential()
generator.add(Dense(units = 7 * 7 * 128, input_shape = (noisedim,)))
generator.add(Reshape((7,7,128)))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization())
generator.add(Conv2DTranspose(64, (3,3), strides=(2,2), padding= "same"))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization())
generator.add(Conv2DTranspose(1, (3,3), strides=(2,2), padding= "same", activation="tanh"))
generator.compile(loss = keras.losses.binary_crossentropy, optimizer = adam)
generator.summary()
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=(3,3), strides=(2,2), padding="same", input_shape= (28,28,1)))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Conv2D(128, kernel_size=(3,3), strides=(2,2), padding="same"))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Flatten())
discriminator.add(Dense(100))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dense(1, activation="sigmoid"))
discriminator.compile(loss = keras.losses.binary_crossentropy, optimizer = adam)
discriminator.summary()
discriminator.trainable = False
ganinput = Input(shape=noisedim, )
genimg = generator(ganinput)
ganoutput = discriminator(genimg)
model = Model(ganinput, ganoutput)
model.compile(loss=keras.losses.binary_crossentropy, optimizer= adam)
model.summary()
xtrain = xtrain.reshape(-1, 28, 28, 1)
xtrain.shape
def displayimages(samples = 25):
    noise = np.random.normal(0,1,size=(samples, noisedim))
    generatedimage = generator.predict(noise)
    plt.figure(figsize=(10, 10))
    for i in range(samples):
        plt.subplot(5,5,i+1)
        plt.imshow(generatedimage[i].reshape(28,28), cmap= "binary")
        plt.axis("off")
    plt.show()
dlosses = []
glosses = []
count = 0
for epoch in range(totalepochs):
    epochdlosses = 0.0
    epochglosses = 0.0
    for step in range(numofbatch):
        discriminator.trainable = True
        idx = np.random.randint(0,60000, halfbatch)
        realimage = xtrain[idx]
        noise = np.random.normal(0,1, size=(halfbatch,noisedim))
        fakeimage = generator.predict(noise)
        realy = np.ones((halfbatch, 1))*0.9
        fakey = np.zeros((halfbatch, 1))
        dlossreal = discriminator.train_on_batch(realimage, realy)
        dlossfake = discriminator.train_on_batch(fakeimage, fakey)
        dloss = 0.5 * dlossreal + 0.5 * dlossfake
        epochdlosses += dloss
        discriminator.trainable = False
        noise = np.random.normal(0,1,size=(batchsize, noisedim))
        groundtruthy = np.ones((batchsize, 1))
        gloss = model.train_on_batch(noise, groundtruthy)
        epochglosses += gloss
        print(f"epoch{epoch + 1}, discriminator losses{epochdlosses/numofbatch}, generator losses{epochglosses/numofbatch}")
        dlosses.append(epochdlosses/numofbatch)
        glosses.append(epochglosses/numofbatch)
        if(epoch + 1) % 10 == 0:
            generator.save("generator.h5")
        displayimages()