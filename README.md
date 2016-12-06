# Distracted-Driver-Detection-with-Deep-Learning
This project aims to detect the dangerous status of driving based on the images captured by the dashboard camera using deep learning.

# Dataset

The dataset is obtained from 

https://www.kaggle.com/c/state-farm-distracted-driver-detection/data

The dataset contains 22,424 images which belongs to one of the 10 classes given below:

    c0: safe driving
    
    c1: texting - right
    
    c2: talking on the phone - right
    
    c3: texting - left
    
    c4: talking on the phone - left
    
    c5: operating the radio
    
    c6: drinking
    
    c7: reaching behind
    
    c8: hair and makeup
    
    c9: talking to passenger
    
We split the data into two sets: training set containing 20,924 images, and validation set containing 1500 images (e.g., 150 images for each class).

# Method 1: train a small Convolutional Neural Network (CNN) from the scatch

Method 1 is implemented in "distracted_driver_detection_with_small_CNN.py". The small CNN consists of 3 convolutional layers with filter size of 3x3, each of which is followed by a max-pooling layer with pool isize of 2x2, and 2 fully-connected dense layers.


# Method 2: train a two-layer dense neural network on top of a pre-trained VGG16 deep CNN

VGG16 is a 16-layer CNN used by the VGG team in the ILSVRC-2014 competition. The VGG16 network structure can be seen in "vgg16_CNN.png". The weights of pre-trained VGG16 CNN can be found at:

https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

Method 2 is implemented in "distracted_driver_detection_with_pretrained_VGG16_deep_CNN.py". The model consists of two parts: the lower part is a pre-trained VGG network with fozen weights, and the upper part is a two-layer dense network. The model is trained using the dataset.  







