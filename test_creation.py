# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:33:01 2022

@author: Yaniv Movshovich
"""

import pathlib
import os
import random
import shutil


def make_test_folder():
    """
    This function makes a test folder if there isn't one already.
    """
    working_dir = pathlib.Path(r'C:\Users\Student\OneDrive\Documents\Deep-Learning-Project-main\‏‏eyesDataset')
    # the path needed to make a test folder in

    os.chdir(working_dir)
    if 'Test' in os.listdir():
        print("Directory Already exists")
    else:
        os.mkdir('Test')


def test_maker(image_count, test_split=0.2):
    """
    This function moves photos from the whole images directory into the test directory using the test split.
    :param image_count: whole dataset images ammount.
    :param test_split: the train test split.

    """
    if test_split == 0:
        return
    real_test_split = test_split / 2
    # splitting the test split into half so half would go to closed eyes and the other half would go to open eyes
    working_dir = pathlib.Path(r"C:\Users\Student\OneDrive\Documents\‏‏eyesDataset2\Images")  # images dataset path
    os.chdir(working_dir)
    random.seed(2)
    classes_names = os.listdir()  # open and closed eyes
    valid_percentage = 0  # this will help printing only integers.
    for photo in range(int(image_count * real_test_split)):
        closed_path = os.path.join(working_dir, classes_names[0])  # closed eyes directory path
        shutil.move(closed_path + '\\' + random.choice(os.listdir(closed_path)),
                    r"C:\Users\Student\OneDrive\Documents\Deep-Learning-Project-main\‏‏eyesDataset\Test\closed_look")
        open_path = os.path.join(working_dir, classes_names[1])  # open eyes directory path
        shutil.move(open_path + '\\' + random.choice(os.listdir(open_path)),
                    r"C:\Users\Student\OneDrive\Documents\Deep-Learning-Project-main\‏‏eyesDataset\Test\open_look")
        work_percentage = float(photo / int(image_count * test_split)) * 200  # percentage of completion
        if work_percentage > valid_percentage:
            print(valid_percentage + 1, '%')
            valid_percentage += 1
        

def back_from_test(data_dir):
    """
    This function brings back the photos from the test directory to the images directory
    :param data_dir: the dataset you want to move back to test
    """
    print(data_dir)
    #path = os.listdir(data_dir)  # the test dataset
    closed_path = pathlib.Path(r"C:\Users\Student\OneDrive\Documents\Deep-Learning-Project-main\‏‏eyesDataset\Images\closed_look")
    test_closed_path = data_dir + "/closed_look"
    test_closed_list = os.listdir(test_closed_path)  
    # closed eyes directory path
    open_path = pathlib.Path(r"C:\Users\Student\OneDrive\Documents\Deep-Learning-Project-main\‏‏eyesDataset\Images\open_look")
    test_open_path = data_dir + "/open_look"
    test_open_list = os.listdir(test_open_path)  
    # open eyes directory path
    for photo in range(len(test_closed_list)):
        shutil.move(test_closed_path + '\\' + test_closed_list[photo], closed_path)
    for photo in range(len(test_open_list)):
        shutil.move(test_open_path + '\\' + test_open_list[photo], open_path)
            
    
#back_from_test(r"D:\Users\yaniv\anaconda3\Projects\DL_Project_Eyes\‏‏eyesDataset\Test")
# delete_similar_images()
# Deleting_Small_photos(r'C:\Users\Student\OneDrive\Documents\eyesDataset2\dataset')
# moving_photos(r"C:\Users\Student\OneDrive\Documents\‏‏eyesDataset2\Test\closed_look")
# test_maker(r"C:\Users\Student\OneDrive\Documents\‏‏eyesDataset2\Images", 15280, 0.1)
# back_to_test(r"C:\Users\Student\OneDrive\Documents\‏‏eyesDataset2\Test\closed_look")
