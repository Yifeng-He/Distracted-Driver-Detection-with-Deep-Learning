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

Method 1 is implemented in distracted_driver_detection_with_small_CNN.py. The small CNN consists of 3 convolutional layers with filter size of 3*3, each of which is followed by a max-pooling layer with pool isize of 2*2, and 2 fully connected layers.




