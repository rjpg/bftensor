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

def get_model_dir(name,erase):
    #base_path = os.path.join(".","dnn")
    model_dir = name #os.path.join(base_path,name)
    os.makedirs(model_dir,exist_ok=True)
    if erase and len(model_dir)>4 and os.path.isdir(model_dir):
        shutil.rmtree(model_dir,ignore_errors=True) # be careful, this deletes everything below the specified path
    return model_dir

tf.logging.set_verbosity(tf.logging.INFO) # ts logging to normal 
#logging.getLogger().setLevel(logging.INFO) # print train evolution

dir_process = pd.read_csv('../directories-to-process.csv',header=None)
np_dir_process = dir_process.as_matrix()

print("Number of models to process : %d" %np_dir_process.shape[0])

#for x in range(0, np_dir_process.shape[0]):
x=208
print("Processing category %d" %np_dir_process[x][0])
#print(np_dir_process[x][1])


base_path=np_dir_process[x][1]
file_path = os.path.join(np_dir_process[x][1],"NNNormalizeData-out.csv")

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

train_inputs, test_inputs, train_output, test_output = train_test_split(total_inputs, total_output, test_size=0.2, random_state=42)

print("Total number of examples %d" %(total_inputs.shape[0]+1))
print("Train number of examples %d" %(train_inputs.shape[0]+1))
print("Test number of examples %d" %(test_inputs.shape[0]+1))

model_dir = get_model_dir(base_path+'/ModelSave',True)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=train_inputs.shape[1])]

training_steps = 40000
save_checkpoints_steps=400 # used to run the early stopping monitor 
early_stopping_rounds=2000   # after 50 SPETS if "loss" doesn't improve stop the train
batch_size=int((train_inputs.shape[0]+1)/20) # train number of examples / 10 - 10 batch = 1 epoch

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
                                 ,config=tf.contrib.learn.RunConfig(save_checkpoints_steps=save_checkpoints_steps
                                                                    ,save_checkpoints_secs=None)
                                 ,feature_columns=feature_columns)


#early stopping using validation monitor 
print("configuring early stopping")
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    x=test_inputs,
    y=test_output,
    #every_n_steps=200,  # when to run the monitor - not working - forcing with save_checkpoints_steps
    early_stopping_metric="accuracy",         #"accuracy" or "loss"
    early_stopping_metric_minimize=False,     #False Maximize accuracy (True is minimize applied to loss)  
    early_stopping_rounds=early_stopping_rounds)

print("Fit will start...")
#classifier = learn.SKCompat(classifier) # For Sklearn compatibility
classifier.fit(train_inputs, train_output, steps=training_steps , 
               batch_size=batch_size,
               monitors=[validation_monitor])
print("Fit is finish...")
 
#print (classifier.get_variable_names()) 


final_model_dir = get_model_dir(base_path+'/ModelSaveFinal',True)
#Save Model into saved_model.pbtxt file (possible to Load in Java)
tfrecord_serving_input_fn = tf.contrib.learn.build_parsing_serving_input_fn(layers.create_feature_spec_for_parsing(feature_columns))  
classifier.export_savedmodel(export_dir_base=final_model_dir, serving_input_fn = tfrecord_serving_input_fn,as_text=True)




    