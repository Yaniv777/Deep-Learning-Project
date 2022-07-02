# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 12:26:19 2022

@author: Yaniv Movshovich
"""

# Import Module
from tkinter import *
from tkinter import filedialog, messagebox
import pathlib
import time
from PIL import ImageTk, Image
from main_form import learning_phase, predicting, model_score_testing



def quit(root):
    root.quit()


def model_selection():
    root = Tk()
    
    models_dir = pathlib.Path(r"C:\Users\Student\OneDrive\Documents\Deep-Learning-Project-main\‏‏eyesDataset\Model")
    root.filename = filedialog.askopenfilename(initialdir=models_dir,
                                               title="Select A Model",
                                               filetypes=(("h5 files", "*.h5"),("all files","*.*")))
    if root.filename.endswith('.h5'):
        global eye_model
        eye_model = root.filename
        success_label = Label(root, text="Great Model!", fg="green").pack()
        
        local_test = messagebox.askquestion("Predict options", "Do you want the model to predict your own photos?")
        if local_test == "yes":
            label = Label(root, text="move the photos you want to test to the test directory with the right label")
            root.filename = filedialog.askopenfilename(
                            initialdir=r"C:\Users\Student\OneDrive\Documents\Deep-Learning-Project-main\‏‏eyesDataset\Test")
        score = model_score_testing(eye_model)
        print (score)
        loss_accuracy = "Test loss: " + str(score[0]) + '\nTest accuracy: ' + str(score[1])
        print (loss_accuracy)
        messagebox.showinfo("Eye Status", loss_accuracy)
        predicting(eye_model)
        quit(root)
            #elif local_test == "no"         
    else:
        failed_label = Label(root, text="Enter A Valid Model! (h5 file)", fg="red").pack()



def split_getter():
    root = Tk()
    split = messagebox.askquestion("model splits", "yes - 70/10/20 \nno 65/15/20")
    if split == "no":
        learning_phase(65,15,20)
    elif split == "yes":
        learning_phase(70,10,20)
    beggining()
'''
    beggining()
    model_split = Label(root, text="Please Enter Train_Validation_Test Split (x,y,z)").pack()
    split = Entry(root, width=50)
    split.pack()
    split.insert(0, "")
    split_button = Button(root, text="Enter Split", command=train_activation)
    print(split.get())
    #learning_phase(split.get())
'''


def beggining():

    root = Tk()
    # root window title and dimension
    root.title("Eyes Status Definer")
    # Set geometry (widthxheight)
    root.geometry('500x500')

    label = Label(root, text="This project can define whether an eye is open or closed\n(needs to be a clear eye picture in order to ensure success)")
    label.pack()
    label = Label(root, text= "Train or Test?", fg="blue")
    label.pack()
    train_or_test = messagebox.askquestion("Predict options",
                                           "yes - if you want to train the model \nno - if you only want to test the model")
    if train_or_test == "no":
        model = Button(root, text="Select Model", command=model_selection, fg="blue")
        model.pack()
    elif train_or_test == "yes":
        train = Button(root, text="Enter Split Rates", command=split_getter, fg="purple")
        train.pack()    
        
    # all widgets will be here
    # Execute Tkinter
    root.mainloop()



    




