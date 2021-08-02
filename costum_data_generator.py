# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:16:21 2019

@author: laakom
"""

import numpy as np
from  tensorflow.keras.utils import Sequence
import cv2

             
class DataGeneratoraug_BoCF(Sequence):
    'Generates data for INTEL-TAU'
    def __init__(self, list_IDs, ground_truth, batch_size=16, dim=(512,512), 
                 n_channels=3,shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.ground_truth = ground_truth
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
       # X, y = self.__data_generation(list_IDs_temp)
        X = (np.array([cv2.imread(ID,-1) for ID in list_IDs_temp ]) *1.0 / 255.0 ).astype('float32')
        # Store class
        y = np.array([self.ground_truth[ID]  for ID in list_IDs_temp],dtype = 'float32' )        
                                            
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
   
   
    
    