'''
Created on 31/07/2018

@author: rjpg
'''


import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten 
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils

import numpy as np
import pandas as pd


from sklearn.cross_validation import train_test_split

from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.engine.topology import Layer,InputSpec


sess = tf.Session()
K.set_session(sess)
K.set_image_dim_ordering("th")

class CylindricalPad(Layer):

    def __init__(self, n=1,m=0, **kwargs):
        super(CylindricalPad, self).__init__(**kwargs)
        self.n = n
        self.m = m
        assert n > 0, 'n must be positive'
        
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(CylindricalPad, self).build(input_shape)  
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                    input_shape[1],
                    input_shape[2] + 2*self.m,
                    input_shape[3] + 2*self.n)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],
                    input_shape[1],
                    input_shape[2] + 2*self.m,
                    input_shape[3] + 2*self.n)
    
    def call(self,testElement, mask=None):
        firstColumns=testElement[:,:,:,0:self.n]
        lastColumns=testElement[:,:,:,testElement.shape[3]-self.n:testElement.shape[3]]
        result=tf.concat([testElement,firstColumns], axis=3)
        result=tf.concat([lastColumns,result], axis=3)
        if self.m != 0 :
            #x = tf.placeholder(result.dtype, shape=[result.shape[0],result.shape[1],self.m,result.shape[3]])
            #y = tf.zeros_like(x,dtype=result.dtype)
            
            firstRows=result[:,:,0:self.m,:]
            
            y = tf.fill(tf.shape(firstRows), 0.)
            
            #y=tf.fill([result.shape[0],result.shape[1],self.m,result.shape[3]],0.)
            
            #y = tf.constant(0., shape=[result.shape[0],result.shape[1],self.m,result.shape[3]],dtype=result.dtype)
            result=tf.concat([y,result], axis=2)
            result=tf.concat([result,y], axis=2)
        
        return result

    def get_config(self):
        config = {'CylindricalPad': self.cropping}
        base_config = super(CylindricalPad, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LeNet:
    @staticmethod
    def build(timeSteps,variables,classes):
        #CONV=>POOL
        inputNet = Input(shape=(1,timeSteps,variables)) 
        conv1=Conv2D(20,kernel_size=(5,2), padding="same")(inputNet)
        conv1=Activation("relu")(conv1)
        conv1=MaxPooling2D(pool_size=(2, 1), strides=(1, 1))(conv1)
        conv2=Conv2D(50,kernel_size=(5,3), padding="same")(conv1)
        conv2=MaxPooling2D(pool_size=(2, 1), strides=(1, 1))(conv2)
        conv2=Activation("relu")(conv2)
        out1=Dropout(0.40)(conv2)
        
        flatten=Flatten()(out1)
        denselayers=Dense(400)(flatten)
        denselayers=Activation("relu")(denselayers)
        denselayers=Dropout(0.5)(denselayers)
        denselayers=Dense(150)(denselayers)
        denselayers=Activation("relu")(denselayers)
        denselayers=Dropout(0.8)(denselayers)
        # a softmax classifier
        classificationLayer=Dense(classes)(denselayers)
        classificationLayer=Activation("softmax")(classificationLayer)
        
        model=Model(inputNet,classificationLayer)
        return model
        
        
class LeNetCylindrical:
    @staticmethod
    def build(timeSteps,variables,classes):
        
        inputNet = Input(shape=(1,timeSteps,variables)) 
        
        cyPad1=CylindricalPad(n=1,m=2)(inputNet)
        conv1=Conv2D(20, (5,3), padding="same")(cyPad1)
        conv1=Activation("relu")(conv1)
       
        cyPad2=CylindricalPad(n=1,m=2)(conv1)
        conv2=Conv2D(50,(5,3), padding="same")(cyPad2)
        
        conv2=Activation("relu")(conv2)
        out1=Dropout(0.40)(conv2)
        flat=Flatten()(out1)
        denselayers=Dense(400)(flat)
        denselayers=Activation("relu")(denselayers)
        denselayers=Dropout(0.5)(denselayers)
        denselayers=Dense(150)(denselayers)
        denselayers=Activation("relu")(denselayers)
        denselayers=Dropout(0.8)(denselayers)
        # a softmax classifier
        classificationLayer=Dense(classes)(denselayers)
        classificationLayer=Activation("softmax")(classificationLayer)
        
        model=Model(inputNet,classificationLayer)
        return model

df = pd.read_csv('../NNNormalizeData-out.csv',header=None) 

np.random.seed(42) # always shuffle the same way 
df = df.reindex(np.random.permutation(df.index)) # shuffle examples 
df.reset_index(inplace=True, drop=True)

print(df.shape) #shape[1] number of columns. [35] is the output column 

inputs = range(df.shape[1]-1) 
target = [df.shape[1]-1] 

total_inputs,total_output = df.as_matrix(inputs).astype(np.float32),df.as_matrix(target).astype(np.int32)
total_inputs = np.reshape(total_inputs, (-1,5,7))
total_inputs=np.transpose(total_inputs,(0,2,1))

modeltest=LeNetCylindrical.build(timeSteps=7,variables=5,classes=5)
modeltest.compile(optimizer=Adam(), loss="categorical_crossentropy",metrics=["accuracy"])
modeltest.summary() 

X_train=total_inputs
y_train=total_output


# network and training
NB_EPOCH = 200
BATCH_SIZE = 64
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT=0.2
NB_CLASSES = 5  # number of outputs = number of classes

print("-------------- [0 , 1] ----------------------")
X_train += 1 
X_train /= 2
print(X_train.shape)
#X_test += 1  

X_train = X_train[:, np.newaxis, :, :]

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, NB_CLASSES)


tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
esCallBack = EarlyStopping(monitor='val_acc', min_delta=0, patience=12, verbose=0, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)

history = modeltest.fit(X_train, y_train, 
        batch_size=BATCH_SIZE, epochs=NB_EPOCH, 
        verbose=1, # 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
        validation_split=VALIDATION_SPLIT, callbacks=[tbCallBack,reduce_lr,esCallBack])

    