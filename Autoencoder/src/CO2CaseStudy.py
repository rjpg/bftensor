'''
Created on 14/03/2018

@author: rjpg
'''

from keras import backend as K
from keras.callbacks import CSVLogger
from pandas import read_csv
from datetime import datetime
from pandas import read_csv
from matplotlib import pyplot
import matplotlib.pyplot as plt
from math import sqrt
from numpy import concatenate
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

#Level 0 : [ 0.0 , 0.0189999988675999960.0]
#Level 1 : [ 0.0189999988675999960.0 , 0.048999997079599990.0]
#Level 2 : [ 0.048999997079599990.0 , 0.089999994635999980.0]
#Level 3 : [ 0.089999994635999980.0 , 0.153999990821599970.0]
#Level 4 : [ 0.153999990821599970.0 , 1.0]

class LeNet:
    @staticmethod
    def build(timeSteps, variables , classes):
        model = Sequential()
        input_shape= (1, timeSteps, variables)
        # CONV => RELU => POOL
        # kernel size (width, height) default (2,5)
        model.add(Conv2D(20, kernel_size=(2,2), padding="same",
            input_shape=input_shape))
        model.add(Activation("relu"))
        # pool size - down scale int factor (vertical, horizontal)
        model.add(MaxPooling2D(pool_size=(2,1),strides=(2, 1)))
        # CONV => RELU => POOL
        
        model.add(Conv2D(50, kernel_size=(3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,1),strides=(2, 1))) #1,2
        #model.summary()
        model.add(Dropout(0.40))
        # Flatten => RELU layers
        model.add(Flatten())
        model.add(Dense(400))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(150))
        model.add(Activation("relu"))
        model.add(Dropout(0.8))
        # a softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        #model.add(Dropout(0.3)) # not logical to do droput on last layer with softmax 
        return model


def fromNumberToClassEncoded(values,intervals):
    
    target = []
    for x in values:
        #print(x)
        for n in range(0,len(intervals)):
            if x > intervals[n][0] and x <= intervals[n][1] :
                target.append(n)
            #    print(n)
            #print(intervals[n][0],"  ", intervals[n][1])
            
    print ("Value to class :\n" , values[32],target[32])
    hotEncoded = np_utils.to_categorical(target, len(intervals))
    print("hot encoded : \n ", hotEncoded[32])
    return hotEncoded ,  target

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# load data
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')
dataset = read_csv('../raw.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
# manually specify column names
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# mark all NA values with 0
dataset['pollution'].fillna(0, inplace=True)
# drop the first 24 hours
dataset = dataset[24:]
# summarize first 5 rows
print(dataset.head(5))
# save to file
dataset.to_csv('../pollution.csv')


# load dataset
dataset = read_csv('../pollution.csv', header=0, index_col=0)
values = dataset.values


# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
# plot each column

#pyplot.figure()
#pyplot.figure(figsize=(20,20))
#for group in groups:
#    pyplot.subplot(len(groups), 1, i)
#    pyplot.plot(values[:, group])
#    pyplot.title(dataset.columns[group], y=0.5, loc='right')
#    i += 1
#plt.savefig("test2.svg", format="svg")
#pyplot.show()


#plt.hist(values[:, 0], bins='auto') 
#plt.title("Histogram of target (unscaled)")
#plt.savefig("test.svg", format="svg")
#plt.show()


# load dataset
dataset = read_csv('../pollution.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# specify the number of lag hours
n_hours = 64
n_features = 8
N_hours_ahead = 1
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, N_hours_ahead)
print(reframed.shape)
print(reframed)


# split into train and test sets
values = reframed.values
n_train_hours = 1* 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

print("train[0] : \n",train[0])
print("train[0].shape : \n",train[0].shape)
print("train : \n",train[0, -n_features])

#################################################################
## SAVE all values of target to conver into classification then
## using histogram in java ...
#################################################################
import numpy
a = numpy.asarray(values[:, -n_features])
print("array : " , a)
numpy.savetxt("../foo.csv", a,fmt='%1.10f', delimiter=",") 


# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

#Level 0 : [ 0.0 , 0.0189999988675999960.0]
#Level 1 : [ 0.0189999988675999960.0 , 0.048999997079599990.0]
#Level 2 : [ 0.048999997079599990.0 , 0.089999994635999980.0]
#Level 3 : [ 0.089999994635999980.0 , 0.153999990821599970.0]
#Level 4 : [ 0.153999990821599970.0 , 1.0]

levels = [[ -1.0 , 0.0189999988675999960],
          [ 0.0189999988675999960 , 0.048999997079599990],
          [ 0.048999997079599990 , 0.089999994635999980],
          [ 0.089999994635999980 , 0.153999990821599970],
          [ 0.153999990821599970 , 1.0]
        ]

print()
train_Y ,values_train = fromNumberToClassEncoded(train_y,levels)
test_Y ,values_test = fromNumberToClassEncoded(test_y,levels)

print("train_y.shape (original)", train_y.shape)
print("train_Y.shape (in classes)", train_Y.shape)


#adapt shape for CNN
X_train = train_X[:, np.newaxis, :, :]

n_hours = 64
n_features = 8

K.set_image_dim_ordering("th")

model = LeNet.build(timeSteps=n_hours,variables=n_features,classes=len(levels))

model.compile(loss="categorical_crossentropy", optimizer=Adam(),metrics=["accuracy"])

model.summary()

NB_EPOCH = 200
# network and training
BATCH_SIZE = 64
VERBOSE = 1
VALIDATION_SPLIT=0.2

# Initialize all variables
#sess.run(tf.global_variables_initializer())


tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
esCallBack = EarlyStopping(monitor='val_acc', min_delta=0, patience=12, verbose=0, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,patience=5, min_lr=0.001)


csv_logger = CSVLogger('log.csv', append=True, separator=',')

history = model.fit(X_train, train_Y, 
        batch_size=BATCH_SIZE, epochs=NB_EPOCH, 
        verbose=1, # 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
        validation_split=VALIDATION_SPLIT, callbacks=[tbCallBack,reduce_lr,esCallBack,csv_logger])

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()