'''
Created on 21/04/2017

@author: birinhos
'''
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import logging
import pandas as pd
import tensorflow.contrib.learn as learn
from sklearn import metrics
from sklearn.cross_validation import train_test_split

#tf output only errors 
tf.logging.set_verbosity(tf.logging.ERROR)

df = pd.read_csv('../NNNormalizeData.csv')  #Numerical output


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

total_inputs,total_output = df.as_matrix(inputs).astype(np.float32),df.as_matrix([target]).astype(np.float32) #important float in output

train_inputs, test_inputs, train_output, test_output = train_test_split(total_inputs, total_output, test_size=0.2, random_state=42)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=train_inputs.shape[1])]
print (feature_columns)
regressor = learn.DNNRegressor(feature_columns=feature_columns,
                                hidden_units=[50, 25, 10])

tf.logging.set_verbosity(tf.logging.INFO) # ts logging to normal 
logging.getLogger().setLevel(logging.INFO) # print train evolution
regressor.fit(train_inputs, train_output,steps=1000)
tf.logging.set_verbosity(tf.logging.ERROR)

# Measure RMSE error.  RMSE is common for regression.
pred = list(regressor.predict(test_inputs, as_iterable=True))
score = np.sqrt(metrics.mean_squared_error(pred,test_output))
print("Test score (RMSE): {}".format(score))

