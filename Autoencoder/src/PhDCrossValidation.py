'''
Created on 28/04/2017

@author: birinhos
'''

from __future__ import division, print_function, absolute_import


import os
import shutil
import tensorflow as tf
import pandas as pd
import tensorflow.contrib.learn as learn
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold

def get_model_dir(name,erase):
    base_path = os.path.join(".","dnn")
    model_dir = os.path.join(base_path,name)
    os.makedirs(model_dir,exist_ok=True)
    if erase and len(model_dir)>4 and os.path.isdir(model_dir):
        shutil.rmtree(model_dir,ignore_errors=True) # be careful, this deletes everything below the specified path
    return model_dir


tf.logging.set_verbosity(tf.logging.INFO)

#save init (?)
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#builder = tf.saved_model.builder.SavedModelBuilder("./model")
#sess.run(tf.global_variables_initializer())

print(tf.VERSION)

df = pd.read_csv('../NNNormalizeData-out.csv')

np.random.seed(42) # always shuffle the same way 
df = df.reindex(np.random.permutation(df.index)) # shuffle examples 
df.reset_index(inplace=True, drop=True)

inputs = []
target = []

y=0;    
for x in df.columns:
    if y != 35 :
        #print("added %d" %y)
        inputs.append(x)
    else :
        target.append(x)
    y+=1

total_inputs,total_output = df.as_matrix(inputs).astype(np.float32),df.as_matrix([target]).astype(np.int32)

# test_inputs/output is the holdout Set
train_inputs, test_inputs, train_output, test_output = train_test_split(total_inputs, total_output, test_size=0.2, random_state=42)


print("Total number of examples %d" %(total_inputs.shape[0]+1))
print("Train number of examples %d" %(train_inputs.shape[0]+1))
print("Test number of examples %d" %(test_inputs.shape[0]+1)) #test data will be used for final evaluation


#global parameters     
training_steps = 1000
save_checkpoints_steps=400 # used to run the early stopping monitor 
early_stopping_rounds=2000   # after 50 SPETS if "loss" doesn't improve stop the train
batch_size=(train_inputs.shape[0]+1)/10 # train number of examples / 10 - 10 batch = 1 epoch

# 5 sets for cross validation applied on train data - test data will be used for final evaluation
kf = KFold(5)

oos_y = []
oos_pred = []
fold = 0
for train, test in kf.split(train_inputs):
    
    fold+=1
    print("Fold #{}".format(fold))
    
    # on train data sub set train/test for cross validation - apply on the inputs and output 
    x_train = train_inputs[train]
    y_train = train_output[train]
    x_test = train_inputs[test]
    y_test = train_output[test]

    batch_size=(x_train.shape[0]+1)/10 # train number of examples / 10 - 10 batch = 1 epoch
    # Get/clear a directory to store the neural network to
    model_dir = get_model_dir('CrossModel-{}'.format(fold),True) # Each fold has its own folder
    
    
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=train_inputs.shape[1])]
    #target_column = [tf.contrib.layers.real_valued_column("output", dimension=train_output.shape[1])]

    classifier = learn.DNNClassifier(hidden_units=[100, 50, 20], n_classes=5
                                 ,model_dir= model_dir
                                 ,config=tf.contrib.learn.RunConfig(save_checkpoints_steps=save_checkpoints_steps
                                                                    ,save_checkpoints_secs=None)
                                 ,feature_columns=feature_columns)


    #early stopping using validation monitor 
    print("configuring early stopping")
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
                x=x_test,
                y=y_test,
                #every_n_steps=200,  # when to run the monitor - not working - forcing with save_checkpoints_steps
                early_stopping_metric="loss",         #"accuracy" of "loss"
                early_stopping_metric_minimize=True,     #False Maximize accuracy (True is minimize applied to loss)  
                early_stopping_rounds=early_stopping_rounds)

    print("Fit will start...")
    #classifier = learn.SKCompat(classifier) # For Sklearn compatibility
    classifier.fit(x_train, y_train, steps=training_steps , 
                   batch_size=batch_size,monitors=[validation_monitor])
    print("Fit is finish...")
 
 

    #Save Model into saved_model.pbtxt file (possible to Load in Java)
    #tfrecord_serving_input_fn = tf.contrib.learn.build_parsing_serving_input_fn(layers.create_feature_spec_for_parsing(feature_columns))  
    #classifier.export_savedmodel(export_dir_base="test", serving_input_fn = tfrecord_serving_input_fn,as_text=True)


# Measure accuracy
pred = list(classifier.predict(test_inputs, as_iterable=True))
#pred = list(classifier.predict(test_inputs))
score = metrics.accuracy_score(test_output, pred)
print("Final score: {}".format(score))

accuracy_score = classifier.evaluate(x=test_inputs,
                                     y=test_output)["accuracy"]

print('Accuracy: {0:f}'.format(accuracy_score))

# test individual samples 
sample_1 = np.array( [[0.37671986791414125,0.28395908337619136,-0.0966095873607713,-1.0,0.06891621389763203,-0.09716678086712205,0.726029084013637,4.984689881073479E-4,-0.30296253267499107,-0.16192917054985334,0.04820256230479658,0.4951319883569152,0.5269983894210499,-0.2560313828048315,-0.3710980821053321,-0.4845867212612598,-0.8647234314469595,-0.6491591208322198,-1.0,-0.5004549422844073,-0.9880910165770813,0.5540293108747256,0.5625990251930839,0.7420121698556554,0.5445551415657979,0.4644276850235627,0.7316976292340245,0.636690006814346,0.16486621649984112,-0.0466018967678159,0.5261100063227044,0.6256168612312738,-0.544295484930702,0.379125782517193,0.6959368575211544]], dtype=float)
sample_2 = np.array( [[1.0,0.7982741870963959,1.0,-0.46270838239235024,0.040320274521029376,0.443451913224413,-1.0,1.0,1.0,-1.0,0.36689718911339564,-0.13577379160035796,-0.5162916256414466,-0.03373651520104648,1.0,1.0,1.0,1.0,0.786999801054777,-0.43856035121103853,-0.8199093927945158,1.0,-1.0,-1.0,-0.1134921695894473,-1.0,0.6420892436196663,0.7871737734493178,1.0,0.6501788845358409,1.0,1.0,1.0,-0.17586627413625022,0.8817194210401085]], dtype=float)

pred = list(classifier.predict(sample_2, as_iterable=True))
print("Prediction for sample_2 is:{} ".format(pred))

pred = list(classifier.predict_proba(sample_2, as_iterable=True))
print("Prediction for sample_2 is:{} ".format(pred))

