from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.estimator import regression

# Data loading
from tflearn.datasets import cifar10
(X, Y), (testX, testY) = cifar10.load_data()
Y = tflearn.data_utils.to_categorical(Y, 10)
testY = tflearn.data_utils.to_categorical(testY, 10)

# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop([32, 32], padding=4)

#Trainning

network = conv_2d(network, 8, 3, activation='relu')
network = conv_2d(network, 8, 3, activation='relu')
network = max_pool_2d(network, 2 ,strides=2)

network = conv_2d(network, 16, 3, activation='relu')
network = conv_2d(network, 16, 3, activation='relu')
network = max_pool_2d(network, 2 ,strides=2)

network = conv_2d(network, 32, 3, activation='relu')
network = conv_2d(network, 32, 3, activation='relu')
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2 ,strides=2)

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2 ,strides=2)

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2 ,strides=2)

network = fully_connected(network, 2056, activation='relu')
network = dropout(network,0.5)
network = fully_connected(network, 2056, activation='relu')
network = dropout(network,0.5)
network = fully_connected(network, 10, activation='softmax')
# Regression

mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
network = tflearn.regression(network, optimizer=mom,
                         loss='categorical_crossentropy',learning_rate=0.001)
# Training
model = tflearn.DNN(network, checkpoint_path='model_resnet_cifar10',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)

model.fit(X, Y, n_epoch=20, validation_set=(testX, testY),
          snapshot_epoch=False, snapshot_step=50,
          show_metric=True, batch_size=128, shuffle=True,
          run_id='resnet_cifar10')
