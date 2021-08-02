#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:42:11 2020

@author: laakom
"""

from tensorflow.keras.layers import Input,Dense, Lambda,BatchNormalization,Activation, multiply,Reshape,Flatten, Conv2D ,Lambda
from tensorflow.keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
import numpy as np
from INTEL_TAU_1080p import INTEL_TAU_DATASET
from costum_data_generator import DataGeneratoraug_BoCF
from BoF_layers import BoF_Pooling, BoF_Pooling_attention_hist,BoF_Pooling_attentionbefore
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, LambdaCallback,CSVLogger
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="0"



def BoCF(input_shape= (227,227,3),n_codewords = 150 , show_summary= False,attention =2) :
    'attention = 0 no attention , 1= before histogram , 2 = after histogram'
    X_input = Input(input_shape)
    
#    # Stage 1
    X = Conv2D(60, (3, 3), strides = (1, 1), kernel_initializer='glorot_uniform',name = 'conv1')(X_input)
    X = MaxPooling2D((8, 8), strides=(2, 2))(X)
    X = Conv2D(30, (3, 3), activation='relu',kernel_initializer='glorot_uniform')(X)
    X = MaxPooling2D((4, 4), strides=(2, 2))(X)
    X = Conv2D(30, (3, 3), activation='relu',kernel_initializer='glorot_uniform')(X)
    if attention ==0 :    
        X =BoF_Pooling(n_codewords, spatial_level=0)(X)
    elif attention ==1:    
        X =BoF_Pooling_attentionbefore(n_codewords, spatial_level=0)(X)
    elif attention ==2:    
        X =BoF_Pooling_attention_hist(n_codewords, spatial_level=0)(X)                
    X = Dropout(rate = 0.2)(X);
    # Stage 2
    X = Dense(40, activation='sigmoid', name='BoC_model_attention')(X);
    
    X = Dropout(rate = 0.2)(X);
    X = Dense(3, activation='relu', name='estimation')(X);    
    X = Lambda(lambda x: K.l2_normalize(x))(X); 
    color_model = Model(inputs = X_input, outputs = X, name='bagofcolors_with_attention');
    if show_summary:
        color_model.summary()
        
    color_model.compile(loss=angluar_error, optimizer=Adam(lr = 0.05))    
    

    
    return color_model












def test_model(model,data,groundtruth,method,path,result_path):        
    from prediction_utilities import save_errors, ill_predict,print_angular_errors,angular_error_recovery,angular_error_reproduction
    print('predicting...') 
    gr, predictions = ill_predict(method,model,data,groundtruth)    
    old_rae = angular_error_recovery(gr, predictions)
    print('recovery errors : \n') 
    print_angular_errors( old_rae)
    save_errors(old_rae,result_path + 'recovery_error')
    new_err = angular_error_reproduction(gr, predictions)
    print('reproduction errors : \n') 
    print_angular_errors(new_err)
    save_errors(new_err,result_path+'reproduction_error')
    return  gr, predictions 



def all_callbacks(path):
    
        
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss',
      		                                            patience=15,
      		                                            factor=0.5,
      		                                            min_lr=0.00005)
    
    		# stop training if no improvements are seen
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=100,restore_best_weights=True)
    
    		# saves model weights to file
    checkpoint=ModelCheckpoint(path , monitor='val_loss', save_best_only=True, save_weights_only=True)
    savecsvlog = CSVLogger(path[:-len('.hdf5')] + '_log.csv', separator=',', append=True )    
        
    return  [learning_rate_reduction,savecsvlog,early_stop,checkpoint] # ,[checkpoint] 

def angluar_error(y_true, y_pred):
   deff =  tf.acos( K.clip(K.sum( tf.nn.l2_normalize(y_true ,axis = -1)*tf.nn.l2_normalize(y_pred ,axis = -1) ,axis=-1,keepdims=True ) , 0.000001, 1.- 0.000001))
   return  deff *180.0 / np.pi   



def get_aug_perfolder(sub_path):
    #read samples
    from glob import glob    
    aug_samples = glob(sub_path + "*.png")
    
    #get_groundtruth
    import pickle
    pkl_file = open( sub_path + 'ground_truth.pkl', 'rb')
    aug_labels = pickle.load(pkl_file)
    pkl_file.close()
    
    return aug_samples, aug_labels

def get_augmentedpartion(partition,ground_truths,aug_path):
    
   #train_data
  train_path =   aug_path + 'train/'
  aug_samples, aug_labels = get_aug_perfolder(train_path)  
  partition['train'] =  aug_samples 
  ground_truths['train'] =  aug_labels
   #validation_data
  val_path =   aug_path + 'val/'
  aug_samples, aug_labels = get_aug_perfolder(val_path)  
  partition['validation'] =  aug_samples 
  ground_truths['validation'] =  aug_labels

  return partition,ground_truths

if __name__ == '__main__':
    
	#change the root_path to the dataset path
    dataset_params = {'camera_names':['Canon_5DSR','Nikon_D810','Sony_IMX135_BLCCSC'],
          'set_names': ['field_3_cameras','field_1_cameras','lab_printouts','lab_realscene'],
          'root_path': '/mnt/Data/Firas2/Intel_v3/processed_1080p'
          }
    #dataset_object
    
    method = 'BoCF'
    inteltau = INTEL_TAU_DATASET(**dataset_params)
    inteltau.set_subsets_splits()
    #for method in methods:
	#train_test only first fold:
    for fold in range(0,1):   
            # Dataset partition: train-validation-test
            partition,ground_truths = inteltau.get_train__test_10folds(fold)            
			#path where to save trained model
            path =  'trained_models/protocol2/train_fold'  + str(fold) + '_' + method 
			
			
		    ##augment the data and save the augmented_dataset in the aug_path:
		    #aug_path data:
            aug_path  = dataset_params['root_path'] + '/expirement_fold_' + str(fold) + '/'
            #check if the augemented data is already created:			
            if os.path.isdir(aug_path):
               print( 'augmented dataset found in ',aug_path ) 
            else:
               print('creating augmented data for fold ', fold) 
               try: 
                   os.mkdir(aug_path)
               except:
                   print("An exception occurred while creating ", aug_path) 
               from utils_data_augmentation import augment_data
               #training data_augmentation:
               train_dir = aug_path + '/train'
               try: 
                   os.mkdir(train_dir)
               except:
                   print("An exception occurred while creating ", train_dir)                
               augment_data(15*len(partition['train']),partition['train'],ground_truths['train'],(227,227),train_dir)    
               #validation data_augmentation:
               val_dir = aug_path + '/val'
               try: 
                   os.mkdir(val_dir)
               except:
                   print("An exception occurred while creating ", val_dir)                
               augment_data(5*len(partition['validation']),partition['validation'],ground_truths['validation'],(227,227),val_dir)  
            
			#creating the partition using the augmented dataset:
            partition,ground_truths = get_augmentedpartion(partition,ground_truths,aug_path)
            
            if method == 'BoCF':

               EPOCHS = 300
			   #create the model:
               model = BoCF(n_codewords = 150 , show_summary= True,attention =2) 
               path = path + 'attention_histogram'
               
				#check if a trained model is there and load the weights if it is true:
               if os.path.isfile(path + '.hdf5'):
                    print('Warning Previous model found. \n  loading weights....')
                    model.load_weights(path + '.hdf5')  
                    # Parameters
             #  else :
                 #dont uncomment this
              #     print('initializing the BoF centers, it can take several mins/hours depending on your machine')
            #       from BoF_layers import initialize_bof_layers
              #     initialize_bof_layers(model, partition['train'],n_samples= 150)
               
			   
			   #train and validation parameters:
            train_params = {'dim': (227,227),
                      'batch_size': 16,
                      'n_channels': 3,
                      'shuffle': True}
            val_params = {'dim': (227,227),
                      'batch_size': 16,
                      'n_channels': 3,
                      'shuffle': True}
            
            print('creating data generators for training with fold', fold )
            training_generator = DataGeneratoraug_BoCF(partition['train'], ground_truths['train'], **train_params)
            validation_generator = DataGeneratoraug_BoCF(partition['validation'], ground_truths['validation'], **val_params)
                                                          
               

            print(method +' training with fold  ', fold )
            history = model.fit_generator(generator=training_generator, epochs=EPOCHS,
                            validation_data=validation_generator,
                            steps_per_epoch = (len(partition['train']) // train_params['batch_size']) ,                    
                            use_multiprocessing=True, 
                            callbacks =all_callbacks( path + '.hdf5' ),
                            workers=4)
            
            

            
            ## test phase:
			#path to save the results in csv:
            result_path =  'trained_models//protocol2//errors//train_fold_' + str(fold)  + '_' + method + 'attention2_bag_of_150'
			#load the model
            model.load_weights(path + '.hdf5') 
            #testing...
            test_model(model,partition['test'],ground_truths['test'],method,path,result_path)

    
    
    