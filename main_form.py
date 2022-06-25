# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 09:32:28 2022

@author: yaniv
"""

#import Eyes_App
import model_testing
import model_learning
import eyes_app


def learning_phase(train, val, test):
    model_learning.model_training(train, val, test)
    
  
def predicting(model_path):
    model_testing.model_predict(model_path)
       
    
def model_score_testing(model_path):
    return model_testing.model_score(model_path)
    
     
def main():
   eyes_app.beggining()
      
    
if __name__ == "__main__":
    main()