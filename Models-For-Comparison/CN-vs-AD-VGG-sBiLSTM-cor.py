print("********************************************************************")
print("CN vs AD VGG sBiLSTM Axial now running")


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
from tensorflow.keras.layers import Dropout, Flatten, Dense,Lambda, Conv2D, MaxPooling2D, LSTM, Input, TimeDistributed, Bidirectional
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


class DataGenerator_CV(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, datapath, batch_size=32, time_steps=86, dim=(32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.time_steps = time_steps
        self.datapath = datapath
        
        
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

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
        
        'Generates data containing batch_size samples' 
        X = np.empty((self.batch_size, self.time_steps, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        
        #print(type(list_IDs_temp))
       
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            label = None
            
            group_dict = self.labels['subject_group']
            slices_dict =  self.labels['subject_slices']
            
            sub_group = group_dict[ID]
            sub_slices = slices_dict[ID]

            if sub_group == 'MCI':
                label = 0
            else:
                label = 1

            # Read slices
            slice_list = []
            for j in sub_slices:
                tmp = np.load(self.datapath + '/' + j)
                slice_list.append(tmp)
            slice_array = np.array(slice_list)
            
            #print(type(i), i)
            #print(slice_array.shape)
            X[i,] = slice_array
            
            # Store class
            y[i] = label

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
    
    df_train_acc = pd.DataFrame(history.history['accuracy'])
    df_train_acc.columns = ['train_accuracy']
    
    df_val_loss = pd.DataFrame(history.history['val_loss'])
    df_val_loss.columns = ['validation_loss']
    
    df_val_acc = pd.DataFrame(history.history['val_accuracy'])
    df_val_acc.columns = ['validation_accuracy']
    
    
    
    df_history = pd.concat([df_train_loss,df_val_loss], axis=1)
    df_history.to_csv(history_path+"/MCI_AD_1365_vgg-bilstm-bilstm_1_16_08_21_axial_slices.csv", index=False)
    return    
    
def fit_generator(model, training_generator, validation_generator, checkpoint_path):    
    cb_save_path = ''
    lrate_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5,verbose=1,min_lr=0.0000001)
    mc = tf.keras.callbacks.ModelCheckpoint('/media/iitindmaths/Seagate_Expansion_Drive/Bup_Backup/SPM/alzheimers-disease/MCI_vs_AD/model_checkpoints/MCI_AD_1365_vgg-bilstm-bilstm_1_16_08_21_axial_slices_{epoch:08d}.h5', 
                                     save_weights_only=True, period=1)
    
    history = model.fit(
                training_generator,
                epochs = 40,
                verbose = 1,
                validation_data = validation_generator,
                callbacks = [lrate_reduce,mc]
            )
    return history 


def fit_cross_validation_triplet(model, cv_file, generator_params_dict, datapath, checkpoint_path, history_path):
    cv_setting = load_cross_validation_settings(cv_file)   
    full_train_dict = cv_setting['train']
    full_val_dict = cv_setting['validation']
    
    train_subject_id = full_train_dict['subject_dict']
    train_subject_group = full_train_dict['subject_group']
    train_subject_slices = full_train_dict['subject_slices']
    train_subject_fnames = list(train_subject_id.keys())
    slice_per_subject = len(train_subject_slices[train_subject_fnames[0]])
    
    val_subject_id = full_val_dict['subject_dict']
    val_subject_group = full_val_dict['subject_group']
    val_subject_slices = full_val_dict['subject_slices']
    val_subject_fnames = list(val_subject_id.keys())
    

    training_generator = DataGenerator_CV(train_subject_fnames, full_train_dict, datapath, **generator_params_dict)


    val_generator_params_dict = generator_params_dict.copy()
    val_generator_params_dict['shuffle'] = False
    validation_generator = DataGenerator_CV(val_subject_fnames, full_val_dict, datapath, **val_generator_params_dict)
    
    
    history = fit_generator(model, training_generator, validation_generator, checkpoint_path)
    save_model_history(history,history_path)
    return        


img_width = 121
img_height = 145
channels = 3
num_slices = 86

datapath = '/media/iitindmaths/Seagate_Expansion_Drive/Bup_Backup/SPM/alzheimers-disease/MCI_vs_AD/npy_large/smwp1'
params_dict = {'dim':(img_width,img_height),
                   'n_channels': 3,
                   'n_classes':2,
                   'batch_size':1,
                   'time_steps':num_slices,
                   'shuffle':True
                  }
history_path = '/media/iitindmaths/Seagate_Expansion_Drive/Bup_Backup/SPM/alzheimers-disease/MCI_vs_AD/history'
checkpoint_path = '/media/iitindmaths/Seagate_Expansion_Drive/Bup_Backup/SPM/alzheimers-disease/MCI_vs_AD/model_checkpoints'
cv_file = '/media/iitindmaths/Seagate_Expansion_Drive/Alz_MCI_vs_AD_1365_each_full_subject_list.pkl'

    

input_tensor = Input(shape=(num_slices,img_width, img_height,channels))

#bilstm-bilstm
vgg = tf.keras.applications.VGG16(include_top=False,input_shape=(img_width, img_height, channels))
vgg_l1 = Conv2D(256, (1,1), padding='same')(vgg.output)
vgg_l2 = Conv2D(128, (1,1), padding='same')(vgg_l1)
vgg_l3 = Conv2D(64, (1,1), padding='same')(vgg_l2)
intermediate_model = Model(inputs=vgg.input, outputs=vgg_l3)
timeDistributed_layer = TimeDistributed( intermediate_model )(input_tensor)
full_model = Model(inputs=input_tensor, outputs=timeDistributed_layer)
flattened_model = TimeDistributed(Flatten())(full_model.output)
lstm = Bidirectional(LSTM(50, return_sequences=True))(flattened_model)
lstm = Bidirectional(LSTM(50,return_sequences=True))(lstm)
lstm = Flatten()(lstm)
dense1 = Dense(10,activation='relu')(lstm)
dense1 = Dense(1,activation='sigmoid')(dense1)
vgg_lstm = Model(inputs=full_model.input, outputs=dense1)

print(vgg_lstm.summary())

# Compile the model
vgg_lstm.compile(
    optimizer=tf.keras.optimizers.Adam(0.000001),
    loss='binary_crossentropy',metrics='accuracy')


print('reached')    
#sys.exit(1)
#fit_cross_validation_triplet(vgg_lstm, cv_file, params_dict, datapath, checkpoint_path, history_path)



history_file = history_path + "/MCI_AD_1365_vgg-bilstm-bilstm_1_16_08_21_axial_slices.csv"

aa = pd.read_csv(history_file)
print(aa['validation_loss'])
print(aa['validation_loss'].min())
print(aa['validation_loss'].idxmin())

min_loss_idx = aa['validation_loss'].idxmin()+1

vgg_lstm.load_weights('/media/iitindmaths/Seagate_Expansion_Drive/Bup_Backup/SPM/alzheimers-disease/MCI_vs_AD/model_checkpoints/MCI_AD_1365_vgg-bilstm-bilstm_1_16_08_21_axial_slices_000000'+str(min_loss_idx)+'.h5')

vgg_lstm.save('CN-vs-AD-VGG-sBiLSTM-cor_MyModel')
#cv_setting = load_cross_validation_settings(cv_file)
#
#print("\n\n\n\n\n\n")
#
#full_test_dict = cv_setting['validation']
#test_subject_id = full_test_dict['subject_dict']
#test_subject_group = full_test_dict['subject_group']
#test_subject_slices = full_test_dict['subject_slices']
#test_subject_fnames = list(test_subject_id.keys())
#slice_per_subject = len(test_subject_slices[test_subject_fnames[0]])
#testing_generator = DataGenerator_CV(test_subject_fnames, full_test_dict, datapath, **params_dict)
#print('validation:\n')
#vgg_lstm.evaluate(testing_generator,verbose=1) #batch size - 4
#
#print("\n\n\n\n\n\n")
#
#full_test_dict = cv_setting['test']
#test_subject_id = full_test_dict['subject_dict']
#test_subject_group = full_test_dict['subject_group']
#test_subject_slices = full_test_dict['subject_slices']
#test_subject_fnames = list(test_subject_id.keys())
#slice_per_subject = len(test_subject_slices[test_subject_fnames[0]])
#testing_generator = DataGenerator_CV(test_subject_fnames, full_test_dict, datapath, **params_dict)
#print('\n\ntesting:\n')
#vgg_lstm.evaluate(testing_generator,verbose=1) #batch size - 4
#
#


