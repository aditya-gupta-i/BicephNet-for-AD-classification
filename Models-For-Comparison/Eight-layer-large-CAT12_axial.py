print("********************************************************************")
print("Eight layer large CAT 12 Axial now running")

import pickle
import pandas as pd
import datetime
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import sys
import random
import tensorflow.keras as keras
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Bidirectional, Lambda, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Input
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import moving_averages
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import initializers

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        #self.class_path = class_path
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
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
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

       
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
        
            
            # Store sample
            tmp = np.load('/media/iitindmaths/Seagate_Expansion_Drive/Bup_Backup/SPM/alzheimers-disease/MCI_vs_AD/npy_large/smwp1/' + ID )
            #tmp_rgb = gray2rgb(tmp)

            X[i,] = tmp
            
            # Store class
            y[i] = self.labels[ID]

        return X, y


    
#returns a list  of  hyperparamter settings
def load_hyperparameter_settings(sampler_file):
    hyp_list = []
    
    with open(sampler_file, "rb") as obj:
        for i in range(10):
            hyp_list.append(pickle.load(obj))
            
    return hyp_list

#returns resNet base_model
def load_base_model_vgg16(img_width,img_height,num_channels,weight_init='imagenet',include_fc_layers=False):
    base_model = applications.VGG16(include_top=include_fc_layers,weights=weight_init, input_shape=(img_width, img_height, num_channels), input_tensor=None, pooling=None)
                                    
    return base_model

# set the first num_layers to nontrainable
# model - an instance of Keras Model
# => model is the final model (base_model added with fully connected layers)

def set_nontrainable_layers(num_layers, model):
    for layer in model.layers[:num_layers]:
        layer.trainable = False
        
    return model

#returns the dict of cross validation settings

def load_cross_validation_settings(cv_file):
    cv_setting = None
    with open(cv_file, "rb") as obj:
        cv_setting = pickle.load(obj)
        
    return cv_setting
    

def save_model_history(history, history_path):    
    df_train_loss = pd.DataFrame(history.history['loss'])
    df_train_loss.columns = ['train_loss']
    df_val_loss = pd.DataFrame(history.history['val_loss'])
    df_val_loss.columns = ['validation_loss']
    df_history = pd.concat([df_train_loss,df_val_loss], axis=1)
    df_history.to_csv(history_path+"/Eight_layer_CLF_MCI_AD_1365_16_08_21_axial_slices.csv", index=False)
    return    
    
def fit_generator(model, training_generator, validation_generator, checkpoint_path):    
    cb_save_path = ''
    lrate_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=2,verbose=1,min_lr=0.0000001)
    mc = tf.keras.callbacks.ModelCheckpoint('/media/iitindmaths/Seagate_Expansion_Drive/Bup_Backup/SPM/alzheimers-disease/MCI_vs_AD/model_checkpoints/Eight_layer_CLF_MCI_AD_1365_16_08_21_axial_slices_{epoch:08d}.h5', 
                                     save_weights_only=True, period=1)
    
    history = model.fit(
                training_generator,
                epochs = 60,
                verbose = 1,
                validation_data = validation_generator,
                callbacks = [lrate_reduce,mc]
            )
    return history 

def fit_crossvalidation(model, cv_file, generator_params_dict, datapath, checkpoint_path, history_path):
    cv_setting = load_cross_validation_settings(cv_file)   
    full_train_dict = cv_setting['train']
    full_val_dict = cv_setting['validation']
    
    train_subject_id = full_train_dict['subject_dict']
    train_subject_group = full_train_dict['subject_group']
    train_subject_slices = full_train_dict['subject_slices']
    train_class_ID = []
    train_class_label = {}
    train_subj_slices = []
        
    for idx in train_subject_id.keys():
        tmp_id = train_subject_id[idx]
        tmp_grp = train_subject_group[idx]
        tmp_slices = train_subject_slices[idx]
        
        label = None
    
        if tmp_grp ==  'MCI':
            label = 0
        else:
            label = 1
            
        for i in tmp_slices:
            train_subj_slices.append(i)
            train_class_label[i] = label
    
    val_subject_id = full_val_dict['subject_dict']
    val_subject_group = full_val_dict['subject_group']
    val_subject_slices = full_val_dict['subject_slices']
    val_class_ID = []
    val_class_label = {}
    val_subj_slices = []
    
    for idx in val_subject_id.keys():
        tmp_id = val_subject_id[idx]
        tmp_grp = val_subject_group[idx]
        tmp_slices = val_subject_slices[idx]

        label = None
    
        if tmp_grp ==  'MCI':
            label = 0
        else:
            label = 1
            
        for i in tmp_slices:
            val_subj_slices.append(i)
            val_class_label[i] = label
        
    training_generator = DataGenerator(train_subj_slices,train_class_label,**generator_params_dict)
        
    val_params_dict = generator_params_dict.copy()
    val_params_dict['shuffle'] = False
    
    validation_generator = DataGenerator(val_subj_slices,val_class_label,**val_params_dict)
    history = fit_generator(model, training_generator, validation_generator, checkpoint_path)
    save_model_history(history,history_path)
    return
    

img_width = 121
img_height = 145
channels = 3

datapath = '/media/iitindmaths/Seagate_Expansion_Drive/Bup_Backup/SPM/alzheimers-disease/MCI_vs_AD/npy_large/smwp1'
params_dict = {'dim':(img_width,img_height),
                   'n_channels': 3,
                   'batch_size': 80,
                   'n_classes': 2,
                   'shuffle':True
                  }
history_path = '/media/iitindmaths/Seagate_Expansion_Drive/Bup_Backup/SPM/alzheimers-disease/MCI_vs_AD/history'
checkpoint_path = '/media/iitindmaths/Seagate_Expansion_Drive/Bup_Backup/SPM/alzheimers-disease/MCI_vs_AD/model_checkpoints'
cv_file = '/media/iitindmaths/Seagate_Expansion_Drive/Alz_MCI_vs_AD_1365_each_full_subject_list.pkl'

model = tf.keras.Sequential([
             tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation=None,padding='same', input_shape=(121,145,3)),
              tf.keras.layers.LeakyReLU(),
             tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=None,padding='same'),
             tf.keras.layers.LeakyReLU(),
             tf.keras.layers.MaxPooling2D(),

             tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation=None,padding='same'),
             tf.keras.layers.LeakyReLU(),
             tf.keras.layers.MaxPooling2D(),

             tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation=None,padding='same'),
             tf.keras.layers.LeakyReLU(),
             tf.keras.layers.MaxPooling2D(),

             tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation=None,padding='same'),
             tf.keras.layers.LeakyReLU(),
             tf.keras.layers.MaxPooling2D(),

             tf.keras.layers.Conv2D(filters=1024, kernel_size=3, activation=None,padding='same'),
             tf.keras.layers.LeakyReLU(),
             tf.keras.layers.MaxPooling2D(pool_size=5),

             tf.keras.layers.Flatten(),
             tf.keras.layers.Dense(100),
             tf.keras.layers.Dense(1, activation='sigmoid')

        ])

print(model.summary())
# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss='binary_crossentropy' ,metrics='accuracy')
print(model.summary())

fit_crossvalidation(model, cv_file, params_dict, datapath, checkpoint_path, history_path)

print('reached')    

history_file = history_path + "/Eight_layer_CLF_MCI_AD_1365_16_08_21_axial_slices.csv"

aa = pd.read_csv(history_file)
print(aa['validation_loss'])
print(aa['validation_loss'].min())
print(aa['validation_loss'].idxmin())

min_loss_idx = aa['validation_loss'].idxmin()+1

model.load_weights('/media/iitindmaths/Seagate_Expansion_Drive/Bup_Backup/SPM/alzheimers-disease/MCI_vs_AD/model_checkpoints/Eight_layer_CLF_MCI_AD_1365_16_08_21_axial_slices_0000000'+str(min_loss_idx)+'.h5')

# TESTING
cv_setting = load_cross_validation_settings(cv_file)
full_test_dict = cv_setting['test']
    
test_subject_id = full_test_dict['subject_dict']
test_subject_group = full_test_dict['subject_group']
test_subject_slices = full_test_dict['subject_slices']
test_subject_fnames = list(test_subject_id.keys())
slice_per_subject = len(test_subject_slices[test_subject_fnames[0]])

pred_list = []
c = 1
for k in test_subject_slices.keys():
    v = test_subject_slices[k]
    pred_array = []
    for i in v:
        ip = np.load('/media/iitindmaths/Seagate_Expansion_Drive/Bup_Backup/SPM/alzheimers-disease/MCI_vs_AD/npy_large/smwp1/' + i)
        #print(ip.shape)
        ip = ip.reshape((1,121,145,3))
        pred = model.predict(ip)
        
        if pred[0] > 0.5:
            val = 1
        else:
            val = 0
        
        pred_array.append(val)
    
    pred_list.append(pred_array)
    print(c)    
    c = c + 1
    

final_pred = []
for pred in pred_list:
    count_1 = pred.count(1)
    count_0 = pred.count(0)

    if count_1 > count_0:
        final_pred.append(1)        
    elif count_1 < count_0:
        final_pred.append(0) 
    else:
        final_pred.append(1) 


label_list = []
for k in test_subject_slices.keys():
    v = test_subject_group[k]

    if v == 'AD':
        label = 1
    else:
        label = 0
    
    label_list.append(label)

Y_true = np.array(label_list)
Y_pred = np.array(final_pred)

print("testing")
print(accuracy_score(Y_true, Y_pred))

# VALIDATION
cv_setting = load_cross_validation_settings(cv_file)
full_val_dict = cv_setting['validation']
    
val_subject_id = full_val_dict['subject_dict']
val_subject_group = full_val_dict['subject_group']
val_subject_slices = full_val_dict['subject_slices']
val_subject_fnames = list(val_subject_id.keys())
slice_per_subject = len(val_subject_slices[val_subject_fnames[0]])

pred_list = []
c = 1
for k in val_subject_slices.keys():
    v = val_subject_slices[k]
    pred_array = []
    for i in v:
        ip = np.load('/media/iitindmaths/Seagate_Expansion_Drive/Bup_Backup/SPM/alzheimers-disease/MCI_vs_AD/npy_large/smwp1/' + i)
        #print(ip.shape)
        ip = ip.reshape((1,121,145,3))
        pred = model.predict(ip)
        
        if pred[0] > 0.5:
            val = 1
        else:
            val = 0
        
        pred_array.append(val)
    
    pred_list.append(pred_array)
    print(c)    
    c = c + 1
    

final_pred = []
for pred in pred_list:
    count_1 = pred.count(1)
    count_0 = pred.count(0)

    if count_1 > count_0:
        final_pred.append(1)        
    elif count_1 < count_0:
        final_pred.append(0) 
    else:
        final_pred.append(1) 

label_list = []
for k in val_subject_slices.keys():
    v = val_subject_group[k]

    if v == 'AD':
        label = 1
    else:
        label = 0
    
    label_list.append(label)

Y_true = np.array(label_list)
Y_pred = np.array(final_pred)

print("validation")
print(accuracy_score(Y_true, Y_pred))
