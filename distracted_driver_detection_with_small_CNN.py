'''
This program is used to detect the driver's status (10 statuses) by using a small convolutional neural network, which
is trained fram scatch using the training images.
'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# training s small convnet from scatch
# convnet: a simple stack of 3 convolution layer with ReLU activation and followed by a max-pooling layers
model=Sequential()
model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3, input_shape=(3,150,150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# the model so far outputs 3D feature maps (height, width, features)
model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64)) # 64 neurons
model.add(Activation('relu'))
model.add(Dropout(0.5)) # drop 50% of neurons

# output layer: classify to 10 driver's states
model.add(Dense(10))
model.add(Activation('softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# the augmentation configuration for generating training data
train_datagen=ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# validation image is scaled by 1/255, no other augmentation on validation data
test_datagen=ImageDataGenerator(rescale=1.0/255)

#this is the generator that will read images found in sub-folders of 'data/train', 
#and indefinitely generate batches of augmented image data
train_generator=train_datagen.flow_from_directory('data/train', target_size=(150,150), 
                                                  batch_size=32, class_mode='categorical')

# this is the  generator for validation data
validation_generator=test_datagen.flow_from_directory('data/validation', target_size=(150,150), 
                                                      batch_size=32, class_mode='categorical')

# train the convolutional neural network
model.fit_generator(train_generator, samples_per_epoch=20924, nb_epoch=20, 
                    validation_data=validation_generator, nb_val_samples=800)

# save the weights
model.save_weights('driver_state_detection_small_CNN.h5')

