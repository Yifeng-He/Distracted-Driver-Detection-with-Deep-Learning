'''
This program is used to recognize the driver's status (one of the 10 statuses) based on the image using pre-trained VGG16 
deep convolutional neural network (CNN).

This program is modified from the blog post: 
"Building powerful image classification models using very little data" from blog.keras.io.

This program do fine tunning for a modified VGG16 net, which consists of two parts: 
the lower model: layer 0-layer24 of the original VGG16 net  (frozen the first 4 blocks, train the weights of the 5-th block 
with our dataset)
the upper model: newly added two layer dense net (train the weights using our dataset)
'''

import os
#The h5py package is a Pythonic interface to the HDF5 binary data format
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense

''' path to the model weights file in HDF5 binary data format
The vgg16 weights can be downloaded from the link below:
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view
'''
weights_path = 'vgg16_weights.h5' 

# dimensions of the images
img_width, img_height = 150, 150

# the path to the training data
train_data_dir = 'data/train'
# the path to the validation data
validation_data_dir = 'data/validation'

# the number of training samples. We have 20924 training images, but actually we can set the 
# number of training samples can be augmented to much more, for example 2*20924
nb_train_samples = 20924

# We actually have 1500 validation samples, which can be augmented to much more
nb_validation_samples = 1500

# number of epoches for training
nb_epoch = 10

# build the VGG16 model
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

'''
# load the weights of the VGG16 networks (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
'''
# load the weights for each layer
assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    # set the weights to layer-k
    model.layers[k].set_weights(weights)
f.close()
print('VGG16 model weights have been successfully loaded.')

# build a MLP classifier model to put on top of the VGG16 model
top_model = Sequential()
# flateen the output of VGG16 model to 2D Numpy matrix (n*D)
top_model.add(Flatten(input_shape=model.output_shape[1:]))
# hidden layer of 256 neurons
top_model.add(Dense(256, activation='relu'))
# add dropout for the dense layer
top_model.add(Dropout(0.5))
# the output layer: we have 10 claases
top_model.add(Dense(10, activation='softmax'))

# connect the two models onto the VGG16 net
model.add(top_model)

# set the first 25 layers (up to the last conv block) of VGFG16 net to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable=False

# compile the model 
model.compile(loss = 'categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# augmentation configuration for training data
train_datagen = ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# augmentation configuration for validation data (actually we did no augmentation to teh validation images)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# training data generator from folder
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width), 
                                                  batch_size=32, class_mode='categorical')

# validation data generator from folder
validation_generator = train_datagen.flow_from_directory(validation_data_dir, target_size=(img_height, img_width), 
                                                       batch_size=32, class_mode='categorical')

# fit the model
model.fit_generator(train_generator, samples_per_epoch=nb_train_samples, nb_epoch=nb_epoch, 
                    validation_data=validation_generator, nb_val_samples=nb_validation_samples)

# save the model weights
# model.save_weights('VGG16_and_MLP_model.h5')

