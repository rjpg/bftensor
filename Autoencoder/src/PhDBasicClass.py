'''
Created on 13/04/2017

@author: birinhos
'''

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import logging
import pandas as pd
import tensorflow.contrib.learn as learn
import os
import shutil
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from tensorflow.python.framework import tensor_shape, graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants



#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#builder = tf.saved_model.builder.SavedModelBuilder("./model")
#sess.run(tf.global_variables_initializer())

print(tf.VERSION)

def get_model_dir(name,erase):
    base_path = os.path.join(".","dnn")
    model_dir = os.path.join(base_path,name)
    os.makedirs(model_dir,exist_ok=True)
    if erase and len(model_dir)>4 and os.path.isdir(model_dir):
        shutil.rmtree(model_dir,ignore_errors=True) # be careful, this deletes everything below the specified path
    return model_dir

#tf output only errors 
tf.logging.set_verbosity(tf.logging.ERROR)

df = pd.read_csv('../NNNormalizeData-out.csv')


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

train_inputs, test_inputs, train_output, test_output = train_test_split(total_inputs, total_output, test_size=0.2, random_state=42)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=train_inputs.shape[1])]
print (feature_columns)

model_dir = get_model_dir('ModelSave',True)

classifier = learn.DNNClassifier(hidden_units=[10, 20, 5], n_classes=5
                                 ,feature_columns=feature_columns
                                 #,optimizer=tf.train.ProximalAdagradOptimizer(
                                 #     learning_rate=0.05,
                                 #     l1_regularization_strength=0.001
                                 #     )
                                 ,model_dir= model_dir #try 1 save
                                 )


tf.logging.set_verbosity(tf.logging.INFO) # ts logging to normal 
logging.getLogger().setLevel(logging.INFO) # print train evolution
classifier.fit(train_inputs, train_output, steps=100)
#back to tf output only errors 
tf.logging.set_verbosity(tf.logging.ERROR)



# Measure accuracy
pred = list(classifier.predict(test_inputs, as_iterable=True))
score = metrics.accuracy_score(test_output, pred)
print("Final score: {}".format(score))

print(pred)
#print(test_output)

#try 2 save
#export_model_dir = get_model_dir('test',True)
#classifier.export(export_dir=model_dir)

#try 3 save
#with tf.Session() as sess:
#    builder = saved_model_builder.SavedModelBuilder("test")
#    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
#    builder.save(True)

#builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
#builder.save(True)

