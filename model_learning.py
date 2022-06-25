# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 08:21:52 2022

@author: Yaniv Movshovich
"""

import matplotlib.pyplot as plt
import numpy as np
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import test_creation
import os


def creating_test_dataset(image_count, test_split):
    """
    this function calls the function that moves photos into the test directory
    Parameters
    ----------
    :param image_count: whole dataset images ammount.
    :param test_split: the train test split.

    Returns
    -------
    None.

    """
    test_creation.test_maker(image_count, test_split)


def model_training(train, val, test):
    """
    this function trains the model, and saves it.
    Parameters
    ----------
    :param train: train split
    :param val: validation split
    :param test: test split

    Returns
    -------
    None.

    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    data_dir = pathlib.Path(r"D:\Users\yaniv\anaconda3\Projects\DL_Project_Eyes\‏‏eyesDataset\Images")  # images dataset
    image_count = len(list(data_dir.glob('*/*.jpeg')))  # dataset images amount
    print(image_count)
    np.random.seed(1671)  # for rep roducibility
    batch_size = 64  # the size of each images batch
    DROPOUT = 0.2  # model dropout
    image_height = 105  # images height
    image_width = 105  # images width
    NB_EPOCH = 25   # amount of times the model will train each image
    test_split = test/100  # how much TRAIN & VALIDATION is reserved for TEST
    
    creating_test_dataset(image_count, test_split)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='binary',
        color_mode='grayscale',
        validation_split=val/100,
        subset="training",
        seed=2,
        image_size=(image_height, image_width),
        batch_size=batch_size
    )
    
    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='binary',
        color_mode='grayscale',
        validation_split=val/100,
        subset="validation",
        seed=2,
        image_size=(image_height, image_width),
        batch_size=batch_size
    )
    
    class_names = train_ds.class_names  # open and closed eyes
    print(class_names)
    plt.figure(figsize=(10, 10))
    
    num_classes = 2  # amount of options (open or closed - 2 options)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    
    model = Sequential([  # the project's model
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(image_height, image_width, 1)),
        layers.Conv2D(32, 1, padding='same', activation='relu'),
        layers.Dropout(DROPOUT),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 1, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 1, padding='same', activation='relu'),
        layers.Dropout(DROPOUT),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(16 , activation='softmax'),
        layers.Dense(num_classes)
        ])

    model.summary()

    
    model.compile(optimizer='Adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    

    history = model.fit(train_ds, validation_data=validation_ds, epochs=NB_EPOCH)

    # the history of accuracies and losses of each epoch throughout the learning phase
    model.save(r"D:\Users\yaniv\anaconda3\Projects\DL_Project_Eyes\‏‏eyesDataset\Model\eyes_define5.h5")
    
    accuracy = history.history['accuracy']  # each epoch train accuracy
    val_acc = history.history['val_accuracy']  # each epoch validation accuracy
    
    loss = history.history['loss']  # each epoch train loss
    val_loss = history.history['val_loss']  # each epoch validation loss
    
    epochs_range = range(NB_EPOCH)  # a list containing numbers from 0 to the number of epochs-1
    plt.figure(figsize=(8, 8))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and validation loss')
    plt.show()
    
    
    class_names = ['C', 'O']  # short classes names for the displays
    plt.figure(figsize=(100, 100))
    
    
