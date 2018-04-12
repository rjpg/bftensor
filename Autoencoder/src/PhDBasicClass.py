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
from nltk.corpus import nps_chat



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
import matplotlib.pyplot as plt
import pandas
#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['indicator 1', 'indicator 2', 'indicator 3', 'indicator 4', 'indicator 5']
#data = pandas.read_csv(url, names=names)
#df.drop(columns=[1, 2,3,4])
for i in range(0,5):
    df=df.drop(df.columns[i], axis=1)
    df=df.drop(df.columns[i], axis=1)
    df=df.drop(df.columns[i], axis=1)
    df=df.drop(df.columns[i], axis=1)
    df=df.drop(df.columns[i], axis=1)
    df=df.drop(df.columns[i], axis=1)
    print (i)

#df=df.drop(df.columns[5], axis=1) #drop output 

df.columns = ['indicator 1', 'indicator 2', 'indicator 3', 'indicator 4', 'indicator 5', 'target']
print(df)




names = ['Strong Down', 'Weak Down', 'neutral', 'Weak UP', 'Strong Up','output']
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(df['target'], facecolor='yellow', edgecolor='gray',bins=5)

# Set the ticks to be at the edges of the bins.
ax.set_xticks(bins)
# Set the xaxis's tick labels to be formatted with 1 decimal place...
from matplotlib.ticker import FormatStrFormatter
ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

# Change the colors of bars at the edges...
twentyfifth, seventyfifth = np.percentile(df['target'], [25, 75])
for patch, rightside, leftside in zip(patches, bins[1:], bins[:-1]):
    if rightside < twentyfifth:
        patch.set_facecolor('green')
    elif leftside > seventyfifth:
        patch.set_facecolor('red')


# Change the colors of bars at the edges...
xx=0
colors = ["red", "orange", "#777777", "lightblue", "blue"]
for patch  in patches:
    patch.set_facecolor(colors[xx])
    xx=xx+1

# Label the raw counts and the percentages below the x-axis...

xx=0;
bin_centers = 0.5 * np.diff(bins) + bins[:-1]
for count, x in zip(counts, bin_centers):
    # Label the raw counts
    ax.annotate(names[xx], xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -18), textcoords='offset points', va='top', ha='center')

    # Label the percentages
    percent = '%0.0f%%' % (100 * float(count) / counts.sum())
    ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -32), textcoords='offset points', va='top', ha='center')
    xx=xx+1

# Give ourselves some more room at the bottom of the plot
plt.subplots_adjust(bottom=0.15)


plt.xticks([])
plt.suptitle("Histogram of Output Classes")
plt.savefig('histo-target-1.pdf')
plt.show()




plt.rcParams["figure.figsize"] = (10,10)
axes =df['target'].plot(kind='hist', subplots=True, layout=(1,1), sharex=False,  alpha=0.5, bins=5, edgecolor="k")#, xlim=(0,210))


colors = ["#e74c3c", "#2ecc71", "#777777", "#3498db", "#3498db"]

for i, ax in enumerate(axes.reshape(-1)):
    # Define a counter to ensure that if we have more than three bars with a value,
    # we don't try to access out-of-range element in colors
    k = 0

    # Optional: remove grid, and top and right spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    
    for rect in ax.patches:
        # If there's a value in the rect and we have defined a color
        if rect.get_height() > 0 and k < len(colors):
            # Set the color
            rect.set_color(colors[k])
            
            # Increment the counter
            k += 1


names = ['indic 1', 'indic 2', 'indic 3', 'indic 4', 'indic 5','output']

plt.subplots_adjust(hspace = 0.5)
plt.suptitle("Histogram of output")
plt.savefig('histo-target.pdf')
plt.show()



s = 121
plt.scatter(df['indicator 1'], df[ 'indicator 2'], alpha=.01 )
plt.xlim(-0.99,0.99)
plt.ylim(-0.99,0.99)
plt.show()

plt.plot(df['indicator 3'])
plt.xlim(-0.0,1000)
plt.show()


correlation = df['indicator 1'].corr(df['indicator 2'])
print("Correlation is: ", correlation)

plt.rcParams["figure.figsize"] = (10,10)

df.plot(kind='hist', orientation='horizontal', stacked=False,subplots=True, layout=(6,1), sharex=False,  alpha=0.5, bins=6)#, xlim=(0,210))
#df.plot(kind='density', subplots=True, layout=(5,1), sharex=False)
#df.hist(layout=(6,1))
plt.subplots_adjust(hspace = 0.5)
plt.suptitle("Histogram of Inputs")
plt.savefig('histo.pdf')
plt.show()

df['indicator 1'].plot(kind='hist', orientation='horizontal', stacked=False,subplots=True, layout=(6,1), sharex=False,  alpha=0.5, bins=6)#, xlim=(0,210))
plt.subplots_adjust(hspace = 0.5)
plt.suptitle("Histogram of Inputs")
plt.savefig('histo.pdf')
plt.show()



names = ['indic 1', 'indic 2', 'indic 3', 'indic 4', 'indic 5','output']
plt.rcParams["figure.figsize"] = (6,6)
correlations = df.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,6,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.subplots_adjust(hspace = 0.5)
plt.suptitle("Correlation Matrix of Inputs")
plt.savefig('Correlation Matrix.pdf')
plt.show()


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

#import matplotlib.pyplot as plt
#plt.hist(df.as_matrix([target], bins='auto') 
#plt.title("Histogram of values of target Class  (equally distributed) ")
#plt.show()

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
#logging.getLogger().setLevel(logging.INFO) # print train evolution
history= classifier.fit(train_inputs, train_output, steps=100)
#back to tf output only errors 
tf.logging.set_verbosity(tf.logging.ERROR)

print(history)

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

