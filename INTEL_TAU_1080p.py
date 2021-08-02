#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:08:13 2019

@author: laakom
"""



import glob
from scipy import io as sio
import numpy as np
from tqdm import tqdm
#import tensorflow as tf
import os
class INTEL_TAU_DATASET:
    
    def __init__(self, camera_names, set_names,root_path):        
            self.camera_names=camera_names
            self.set_names=set_names
            self.root_path = root_path
            self.train_cam = None
            self.splits = None

          
    def get_train_val_test_percamera_ids( self,train_cam = 'Canon_5DSR'):
        "Return a dicionary of the (train,validation,test) split and their ground truth"
        self.train_cam = train_cam
        partition = {}
        ground_truths = {}
        print('generating data for  training with camera:  ', train_cam)
        image_path = self.root_path + train_cam + '//'
        [train_ids, train_ground_truths]= self.generate_data(image_path)        
        assert(len(train_ids) == len(train_ground_truths))
        partition['train'] = train_ids
        ground_truths['train'] = train_ground_truths
      
        #get the validation camera name
        val_cam_index = (self.camera_names.index(self.train_cam)+1) % 3
        val_cam = self.camera_names[val_cam_index]  
        image_path = self.root_path + val_cam + '//'
        #generate the validation images and the ground truth
        [val_ids, val_ground_truths]=self.generate_data(image_path)
        assert(len(val_ids) == len(val_ground_truths))
        partition['validation'] = val_ids
        ground_truths['validation'] = val_ground_truths
           
       
        #get the test camera name
        test_cam_index = (self.camera_names.index(self.train_cam)+2) % 3
        test_cam = self.camera_names[test_cam_index]        
        #create the original images path 
        image_path = self.root_path + test_cam + '//'
        #generate the test images ids and the ground truth dictionary
        [test_ids, test_ground_truths]=self.generate_data(image_path)
        assert(len(test_ids) == len(test_ground_truths))        
        partition['test'] = test_ids    
        ground_truths['test'] = test_ground_truths
        
        return partition,ground_truths
        
         
    def generate_data(self,image_path, set_names=['field_3_cameras','field_1_cameras','lab_printouts','lab_realscene'] ):    
        ids = []
        ground_truths = {}
        for current_set in  set_names:
           files = glob.glob( image_path + current_set + '//' + '*.tiff')
           for file in files:
              ids.append(os.path.realpath(file))
              ground_truths[os.path.realpath(file)]= self.get_ground_truth(file)        
        return ids,ground_truths      
        
        
    def get_ground_truth(self,file):
           'read the ground truth file corresponding to an image file'
           groundtruthfile = file[:-len('.tiff')] 
           #we use the same format of the subset files and the ground truth files:
           index = groundtruthfile.rfind('_')
           groundtruthfile = groundtruthfile[:index+1] + str( int(groundtruthfile[index+1:])).zfill(3) + '.wp'
           gt =[]
           with open(groundtruthfile, "r") as f:
                groundtruth = f.read().split()
           for x in groundtruth:
               gt.append(float(x))
           #the ground truth is saved in  [R,G,B ]form   
           ground_truth = np.asarray(gt)

           return ground_truth/  np.linalg.norm(ground_truth,2)
               
    def read_ccm(self,camera_1,camera_2,root_path):
        #camera_1 is the reference camera
        #camera_2 is the source camera
        #reading the mat files containing the CCM matrices
        ccm_folder = root_path + 'info/Info//'
        ccm = sio.loadmat(ccm_folder + 'CCMmatrices.mat')
        ccm = ccm['CCMat']     
        #getting the indices of the CCM matrice cell, e.g., if want to get the CCM 
        #to trasform nikon to canon that is saved in ccm[1][0]: 1 is the index of nikon
        #and 0 is the index of canon in 'camera_names' set.
        indx1 = self.camera_names.index(camera_1)
        indx2 = self.camera_names.index(camera_2)     
        return np.float64(ccm[indx2,indx1])

    
    def set_subsets_splits(self) :
        import os
        dir = self.root_path + '/splits'
        splits = {}
        for i in tqdm(range(10)):
            file = dir + '//Intel-TAU_v3_filelist_subset' + str(i + 1).zfill(2) + '.txt'
            gt = []
            with open(file, "r") as f:
                curentfile = f.read().split()
            for x in curentfile:
                temp = x[ len('y:\Intel-TAU_v3'): - len('.plain16')] + '.tiff'
                gt.append(self.root_path+ temp.replace('\\' , os.sep))
            splits[str(i + 1).zfill(2)] = gt
        self.splits =   splits    
    
    
    
    def get_groundtruthfold(self,images) :
        ids = []
        ground_truths = {}
        for current_image in  images:           
           #we use the same format of the subset files and the ground truth files:
           image_file =current_image[:-len('.tiff')]  
           index = image_file.rfind('_')
           imagefile = image_file[:index+1] + str( int(image_file[index+1:])).zfill(3) + '.tiff'         
           ids.append(imagefile)
           ground_truths[imagefile]= self.get_ground_truth(current_image)        
        return ids,ground_truths
        
        
        
    
    def get_train__test_10folds( self,fold =0):
        "Return a dicionary of the (train,validation,test) split and their ground truth"
        #self.train_cam = train_cam
        partition = {}
        ground_truths = {}        
        partition['train'] = []
        for current in range(10):
            if current ==fold:
                partition['test'] = self.splits[str(current + 1).zfill(2)]
            else:
                partition['train'].extend( self.splits[str(current + 1).zfill(2)])
            
        
        from sklearn.model_selection import train_test_split
        x_train, x_val = train_test_split(partition['train'], test_size=int(len(partition['train'])*0.25))
        partition['train']= x_train
        partition['validation']= x_val
              
        [train_ids, train_ground_truths]= self.get_groundtruthfold( partition['train'])            
    
        assert(len(train_ids) == len(train_ground_truths))
        partition['train'] = train_ids
        ground_truths['train'] = train_ground_truths
      
        [val_ids, val_ground_truths]= self.get_groundtruthfold( partition['validation'])            
    
        assert(len(train_ids) == len(train_ground_truths))
        partition['validation'] = val_ids
        ground_truths['validation'] = val_ground_truths
      

           
     
        [test_ids, test_ground_truths]=self.get_groundtruthfold( partition['test']) 
        assert(len(test_ids) == len(test_ground_truths))        
        partition['test'] = test_ids    
        ground_truths['test'] = test_ground_truths
        
        return partition,ground_truths



















































































































