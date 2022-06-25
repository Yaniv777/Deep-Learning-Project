# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 12:35:30 2022

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
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

image_height = 105  # images height
image_width = 105  # images width




def model_predict(model_path):
    """
    this function makes the model predict each image whether it is opened or closed

    Parameters
    ----------
    :param model_path: the path of the model we want to test

    Returns
    -------
    None.

    """
    model_path = pathlib.Path(model_path)
    model = tf.keras.models.load_model(model_path)
    model.summary()
    class_names = ['Closed', 'Open']  # short classes names for the displays
    plt.figure(figsize=(100, 100))
    

    open_dir = pathlib.Path(r"D:\Users\yaniv\anaconda3\Projects\DL_Project_Eyes\‏‏eyesDataset\Test\open_look")
    # open eyes test directory path
    open_eyes_paths = list(open_dir.glob('*.jpeg'))  # list of the images in open_dir

    closed_dir = pathlib.Path(r"D:\Users\yaniv\anaconda3\Projects\DL_Project_Eyes\‏‏eyesDataset\Test\closed_look")
    # closed eyes test directory path
    eyes_paths = list(closed_dir.glob('*.jpeg'))  # list of the images in closed_dir
    
    for i in range(len(open_eyes_paths)):
        eyes_paths.append(open_eyes_paths[i])
        

    valid_percentage = 0  # this will help printing only integers.

    for i, eyes_path in enumerate(eyes_paths):
        eye_img = keras.preprocessing.image.load_img(eyes_path, target_size=(105, 105), color_mode='grayscale')
        # a single eye image
        img_arr = keras.preprocessing.image.img_to_array(eye_img)  # 
        img_arr = tf.expand_dims(img_arr, 0)
        
        predictions = model.predict(img_arr)  # model prediction to a certain image in the dataset
        
        score = tf.nn.softmax(predictions[0])  # image prediction score
        #percent = int(round(100 * np.max(score), 0))  # prediction success rate
        title = class_names[np.argmax(score)]
        # image title with class prediction name and success rate
        work_percentage = float(i / len(eyes_paths)) * 100  # percentage of completion
        if work_percentage > valid_percentage:
            print(valid_percentage + 1, '%')
            valid_percentage += 1
        if i <= 7999:
            ax = plt.subplot(100, 80, i + 1)
        else:
            break
        plt.imshow(eye_img)
        plt.title(title)
        plt.axis("off")
        
        
#model_predict(r"D:\Users\yaniv\anaconda3\Projects\DL_Project_Eyes\‏‏eyesDataset\Model\eyes_definer.h5")

def model_score(model_path):
    """
    this function returns the model's score on the test dataset.

    Parameters
    ----------
    :param model_path: the path of the model we want to test

    Returns
    -------
    score: test accuracy and loss

    """
    model_path = pathlib.Path(model_path)
    model = tf.keras.models.load_model(model_path)
    test_dir = pathlib.Path(r"D:\Users\yaniv\anaconda3\Projects\DL_Project_Eyes\‏‏eyesDataset\Test")  # test dataset path
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='grayscale',
        batch_size=489,
        image_size=(image_height, image_width),
        shuffle=True,
        seed=2,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False
        )
    (x_test, y_test) = test_ds  # test dataset with labels
    test_dataset = test_ds.from_tensor_slices((x_test, y_test))
    
    score = model.evaluate(
        x=test_ds,
        batch_size=None,
        verbose="auto",
        sample_weight=None,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        return_dict=False,
        )
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score
    
