# -*- coding: utf-8 -*-
"""
Created on Sun May  1 11:29:05 2022

@author: Yaniv Movshovich
"""

import imagehash
import os
import pathlib
from PIL import Image
import shutil

def similarity_check(photo1, photo2):
    """
    This function checks between two photos how similar their hashes are.
    if the are similar enough, True would be returned.
    I made this function in order to prevent overfitting.
    :param photo1: first photo
    :param photo2: second photo
    :return: True if the photos are similar, False if they aren't
    """
    working_dir = pathlib.Path(r"C:\Users\Student\OneDrive\Documents\‏‏eyesDataset2\Images\closed_look")
    # the directory the images are at
    os.chdir(working_dir)
    hash0 = imagehash.average_hash(Image.open(photo1))  # first image hash
    hash1 = imagehash.average_hash(Image.open(photo2))  # second image hash
    cutoff = 10  # maximum bits that could be different between the hashes. 

    if hash0 - hash1 < cutoff:
        return True
    else:
        return False


def remove_margins(data_dir, margin):
    dataset = os.listdir(data_dir)
    for photo in range(margin):
        shutil.move(data_dir + '//' + dataset[photo],
                    r'C:\Users\Student\OneDrive\Documents\Deep-Learning-Project-main\‏‏eyesDataset\Trash')



def back_from_trash():
    """
    This function brings back the photos from the test directory to the images directory
    :param data_dir: the dataset you want to move back to test
    """
    data_dir = r"C:\Users\Student\OneDrive\Documents\Deep-Learning-Project-main\‏‏eyesDataset\Trash"
    data_dir_list = os.listdir(data_dir)  
    #path = os.listdir(data_dir)  # the test dataset
    closed_path = pathlib.Path(r"C:\Users\Student\OneDrive\Documents\‏‏eyesDataset2\Images\closed_look")
    # closed eyes directory path
    open_path = pathlib.Path(r"C:\Users\Student\OneDrive\Documents\‏‏eyesDataset2\Images\open_look")
    # open eyes directory path
    for photo in range(len(data_dir_list)):
        if data_dir_list[photo].startswith("C"):
            shutil.move(data_dir + '//' + data_dir_list[photo], closed_path)
        elif data_dir_list[photo].startswith("O"):
            shutil.move(data_dir + '//' + data_dir_list[photo], open_path)
    
    
    
    

def delete_similar_images(data_dir):
    """
    This function deletes a photo if there are another photo similar to it.
    """
    working_dir = pathlib.Path(data_dir)  # cwd
    data_dir = os.listdir(working_dir)  # a list full of images taken from the dataset we want to delete images from
    os.chdir(working_dir)
    len_dir = len(data_dir)  # amount of images in the dataset
    valid_percentage = 0  # this will help printing only integers.
    for photo in range(len(data_dir)-1):
        # loops through the whole dataset and deletes the first image if it is similar enough to the other
        if similarity_check(data_dir[photo], data_dir[photo + 1]):
            os.remove(data_dir[photo])
        work_percentage = float(photo / int(len_dir)) * 100  # percentage of completion
        if work_percentage > valid_percentage:
            if (valid_percentage + 1) % 5 == 0:
                print(valid_percentage + 1, '%')
            valid_percentage += 1


def deleting_small_photos(data_dir):
    """
    This function deletes photos if their size is too small
    :param data_dir: the directory you want to delete photos from.
    """
    print(data_dir)
    path = os.listdir(data_dir)  # a list full of images taken from the dataset we want to delete images from
    valid_percentage = 0  # this will help printing only integers.
    for photo in range(len(path)):
        size = os.path.getsize(os.path.join(data_dir, path[photo]))  # certain image file size
        if size < 1800:
            os.remove(os.path.join(data_dir, path[photo]))
        work_percentage = float(photo / len(path)) * 100  # percentage of completion
        if work_percentage > valid_percentage:
            print(valid_percentage + 1, '%')
            valid_percentage += 1


back_from_trash()