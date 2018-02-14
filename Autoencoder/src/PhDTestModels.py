'''
Created on 11/05/2017

@author: rjpg
'''


from __future__ import division, print_function, absolute_import


import os
import logging
import shutil
import tensorflow as tf
import pandas as pd
import tensorflow.contrib.learn as learn
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from tensorflow.contrib import layers


tf.logging.set_verbosity(tf.logging.INFO) # ts logging to normal 
#logging.getLogger().setLevel(logging.INFO) # print train evolution

dir_process = pd.read_csv('../directories-to-process-test.csv',header=None)
np_dir_process = dir_process.as_matrix()

print("Number of models to process : %d" %np_dir_process.shape[0])

#for x in range(0, np_dir_process.shape[0]):
x=210
print("Processing category %d" %np_dir_process[x][0])
#print(np_dir_process[x][1])


base_path=np_dir_process[x][1]
file_path = os.path.join(np_dir_process[x][1],"NNNTestNormalizeData-out.csv")

print(base_path)
print(file_path)

#Read training file 
df = pd.read_csv(file_path,header=None)

np.random.seed(42) # always shuffle the same way 
df = df.reindex(np.random.permutation(df.index)) # shuffle examples 
df.reset_index(inplace=True, drop=True)

inputs = []
target = []

y=0;    
for i in df.columns:
    if y != 35 :
        #print("added %d" %y)
        inputs.append(i)
    else :
        target.append(i)
    y+=1

total_inputs,total_output = df.as_matrix(inputs).astype(np.float32),df.as_matrix([target]).astype(np.int32)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=total_inputs.shape[1])]

print("Total number of examples %d" %(total_inputs.shape[0]+1))

model_dir = base_path+'/ModelSave'


classifier = learn.DNNClassifier(hidden_units=[100, 50, 20], n_classes=5
                                 ,model_dir= model_dir
                                 #,activation_fn=tf.nn.relu
                                 ,optimizer=tf.train.AdadeltaOptimizer(0.5)
                                 
                                 #,optimizer=tf.train.AdagradOptimizer(0.1)
                                 #tf.train.RMSPropOptimizer(learning_rate=0.1) #
                                 #tf.train.ProximalAdagradOptimizer(
                                 #     learning_rate=0.01)
                                 #     l1_regularization_strength=0.001
                                 #     )
                                 #,config=tf.contrib.learn.RunConfig(save_checkpoints_steps=save_checkpoints_steps
                                 #                                   ,save_checkpoints_secs=None)
                                 ,feature_columns=feature_columns)


# Measure accuracy
pred = list(classifier.predict(total_inputs, as_iterable=True))
score = metrics.accuracy_score(total_output, pred)
print("Accuarcy after load: {}".format(score))


# test individual samples 
sample_1 = np.array( [[0.37671986791414125,0.28395908337619136,-0.0966095873607713,-1.0,0.06891621389763203,-0.09716678086712205,0.726029084013637,4.984689881073479E-4,-0.30296253267499107,-0.16192917054985334,0.04820256230479658,0.4951319883569152,0.5269983894210499,-0.2560313828048315,-0.3710980821053321,-0.4845867212612598,-0.8647234314469595,-0.6491591208322198,-1.0,-0.5004549422844073,-0.9880910165770813,0.5540293108747256,0.5625990251930839,0.7420121698556554,0.5445551415657979,0.4644276850235627,0.7316976292340245,0.636690006814346,0.16486621649984112,-0.0466018967678159,0.5261100063227044,0.6256168612312738,-0.544295484930702,0.379125782517193,0.6959368575211544]], dtype=float)
sample_2 = np.array( [[1.0,0.7982741870963959,1.0,-0.46270838239235024,0.040320274521029376,0.443451913224413,-1.0,1.0,1.0,-1.0,0.36689718911339564,-0.13577379160035796,-0.5162916256414466,-0.03373651520104648,1.0,1.0,1.0,1.0,0.786999801054777,-0.43856035121103853,-0.8199093927945158,1.0,-1.0,-1.0,-0.1134921695894473,-1.0,0.6420892436196663,0.7871737734493178,1.0,0.6501788845358409,1.0,1.0,1.0,-0.17586627413625022,0.8817194210401085]], dtype=float)

pred = list(classifier.predict(sample_2, as_iterable=True))
print("Prediction for sample_2 is:{} ".format(pred))

pred = list(classifier.predict_proba(sample_2, as_iterable=True))
print("Prediction for sample_2 is:{} ".format(pred))





