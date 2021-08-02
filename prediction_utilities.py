#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 13:52:52 2019

@author: laakom
"""


import numpy as np
import cv2
from sklearn.feature_extraction import image
from tqdm import tqdm
#from keras.applications.imagenet_utils import preprocess_input
#from utils_viz import adjust_gamma_training




def ill_predict(method,model,testsamplesID,groundtruth_dic):
    
    if method == 'Bianco':
       ground_truths, precitions = Bianco_CNN_predict(model, testsamplesID,groundtruth_dic)
    if method == 'FC4':
       ground_truths, precitions = FC4_predict(model, testsamplesID,groundtruth_dic)
    if method == 'BoCF':
       ground_truths, precitions = FoCF_predict(model, testsamplesID,groundtruth_dic)
    return ground_truths,precitions

def Bianco_CNN_predict(model, testsamplesID,groundtruth_dic):
    
    predictions = []
    ground_truths = []
    for ID in tqdm(testsamplesID):
      img = (cv2.resize(cv2.imread(ID,-1), (0,0), fx=0.25, fy=0.25) *1.0  / 65536.0).astype('float32')
      img_patches = image.extract_patches_2d(img,(32,32), 8)
      img_patches =  (img_patches ).astype('float32')
      patches_prediction = np.mean(model.predict(img_patches),axis=0) 
      patches_prediction = np.clip(patches_prediction, 10**(-6), np.max(patches_prediction) )
      patches_prediction /= np.linalg.norm(patches_prediction,2)
      
      predictions.append(patches_prediction)
      ground_truths.append(groundtruth_dic[ID])
    return np.array(ground_truths,dtype = 'float32' ), np.array(predictions)    






def FoCF_predict(model, testsamplesID,groundtruth_dic):    
    predictions = []
    ground_truths = []
    for ID in tqdm(testsamplesID):
      img = cv2.resize(cv2.imread(ID,-1),(227,227)) 
      img =  (img * 1.0/ 65535.0 ).astype('float32')
      #img =adjust_gamma_training(img)
      img = np.expand_dims(img, axis=0)
      #img =preprocess_input(img)
      #img_patches = preprocess_input(np.float32(img[..., ::-1]*255.0))
      patches_prediction = model.predict(img) 
      patches_prediction = np.clip(patches_prediction[0], 10**(-6), np.max(patches_prediction[0]) )
      patches_prediction /= np.linalg.norm(patches_prediction,2)
      predictions.append(patches_prediction)
      ground_truths.append(groundtruth_dic[ID])
    return np.array(ground_truths,dtype = 'float32' ), np.array(predictions)    







def FC4_predict(model, testsamplesID,groundtruth_dic):    
    predictions = []
    ground_truths = []
    for ID in tqdm(testsamplesID):
      img = cv2.resize(cv2.imread(ID,-1),(227,227)) 
      img =  (img * 1.0/ 65535.0 ).astype('float32')
      #img =adjust_gamma_training(img)
      img = np.expand_dims(img, axis=0)
      #img =preprocess_input(img)
      #img_patches = preprocess_input(np.float32(img[..., ::-1]*255.0))
      patches_prediction = model.predict(img) 
      patches_prediction = np.clip(patches_prediction[0], 10**(-6), np.max(patches_prediction[0]) )
      patches_prediction /= np.linalg.norm(patches_prediction,2)
      predictions.append(patches_prediction)
      ground_truths.append(groundtruth_dic[ID])
    return np.array(ground_truths,dtype = 'float32' ), np.array(predictions)    





def angular_error_reproduction(ground_truth, prediction):
    """
    calculate angular error(s) between the ground truth RGB triplet(s) and the predicted one(s)
    :param ground_truth: N*3 or 1*3 Numpy array, each row for one ground truth triplet
    :param prediction: N*3 Numpy array, each row for one predicted triplet
    :return: angular error(s) in degree as Numpy array
    """
    
    res = np.divide(ground_truth,prediction) 
    res_norm  = res / np.linalg.norm(res, ord=2, axis=-1, keepdims=True)
    u = np.ones(np.shape(res)) / np.sqrt(3)
    u_norm  = u / np.linalg.norm(u, ord=2, axis=-1, keepdims=True)

    return 180 * np.arccos(np.sum(res_norm * u_norm, axis=-1)) / np.pi



def angular_error_recovery(ground_truth, prediction):
    """
    calculate angular error(s) between the ground truth RGB triplet(s) and the predicted one(s)
    :param ground_truth: N*3 or 1*3 Numpy array, each row for one ground truth triplet
    :param prediction: N*3 Numpy array, each row for one predicted triplet
    :return: angular error(s) in degree as Numpy array
    """
    ground_truth_norm = ground_truth / np.linalg.norm(ground_truth, ord=2, axis=-1, keepdims=True)
    prediction_norm = prediction / np.linalg.norm(prediction, ord=2, axis=-1, keepdims=True)
    return 180 * np.arccos(np.sum(ground_truth_norm * prediction_norm, axis=-1)) / np.pi


def summary_angular_errors(errors):
  errors = sorted(errors)

  def g(f):
    return np.percentile(errors, f * 100)

  median = g(0.5)
  mean = np.mean(errors)
  max = np.max(errors)
  trimean = 0.25 * (g(0.25) + 2 * g(0.5) + g(0.75))
  results = {
      '25': np.mean(errors[:int(0.25 * len(errors))]),
      '75': np.mean(errors[int(0.75 * len(errors)):]),
      '95': g(0.95),
      'tri': trimean,
      'med': median,
      'mean': mean,
      'max': max
      
  }
  return results


def just_print_angular_errors(results):
  print("25: %5.3f," % results['25'], end=' ')
  print("med: %5.3f" % results['med'], end=' ')
  print("tri: %5.3f" % results['tri'], end=' ')
  print("avg: %5.3f" % results['mean'], end=' ')
  print("75: %5.3f" % results['75'], end=' ')
  print("95: %5.3f" % results['95'], end=' ')
  print("max: %5.3f" % results['max'])


def print_angular_errors(errors):
  print("%d images tested. Results:" % len(errors))
  results = summary_angular_errors(errors)
  just_print_angular_errors(results)
  


def save_errors(errors,path):
  import csv
  print("%d images tested. Results:" % len(errors))
  results = summary_angular_errors(errors)
  w = csv.writer(open(path+ '.csv', "w"))
  for key, val in results.items():
    w.writerow([key, val])
  just_print_angular_errors(results)