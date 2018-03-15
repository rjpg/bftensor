# import the necessary packages
import tensorflow as tf
#import pydot
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Lambda, Dropout 
from keras.models import Sequential
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Reshape
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam, Adadelta
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.layers.recurrent import LSTM
from keras.backend.tensorflow_backend import reshape
from keras.utils.vis_utils import plot_model
from keras.layers.merge import concatenate
from keras.legacy.layers import Merge, merge
from keras.layers.wrappers import Bidirectional, TimeDistributed
np.random.seed(1671)  # for reproducibility

sess = tf.Session()
K.set_session(sess)

class LSTMNet:
    @staticmethod
    def build(timeSteps,variables,classes):
        #CONV=>POOL
        inputNet = Input(shape=(timeSteps,variables),batch_shape=(20, 7, 5)) 
        #lstm=Bidirectional(LSTM(100,recurrent_dropout=0.4,dropout=0.4),merge_mode='concat')(inputNet) #worse using stateful=True
        lstm=Bidirectional(LSTM(50),merge_mode='concat')(inputNet) #worse using stateful=True 
        #denselayers=Dense(400)(lstm)
        #denselayers=Activation("relu")(denselayers)
        #denselayers=Dropout(0.5)(denselayers)
        #denselayers=Dense(150)(denselayers)
        #denselayers=Activation("relu")(denselayers)
        #denselayers=Dropout(0.8)(denselayers)
        # a softmax classifier
        classificationLayer=Dense(classes,activation='softmax')(lstm)
        #classificationLayer=Activation("softmax")(classificationLayer)
        
        model=Model(inputNet,classificationLayer)
        return model

def independentFilters(element,filterIndex):
    print(element.shape)
    return element[:,filterIndex,:,:]

#define the convnet multi LSTMs
class RJPGNet:
    @staticmethod
    def build(timeSteps,variables,classes,nlstms):
        #CONV=>POOL
        inputNet = Input(shape=(1,timeSteps,variables))#,batch_shape=(20,1, 7, 5)) 
        conv1=Conv2D(20, (2,5), padding="same")(inputNet)
        conv1=Activation("relu")(conv1)
        conv1=AveragePooling2D(pool_size=(2, 1), strides=(1, 1))(conv1)
        conv2=Conv2D(nlstms,(3,3), padding="same")(conv1)
        conv2=AveragePooling2D(pool_size=(2, 1), strides=(1, 1))(conv2)
        conv2=Activation("relu")(conv2)
        channels=Dropout(0.40)(conv2)
        lstmsVec=[]
        
            
        for x in range(0,nlstms):
            filterImg=Lambda(lambda element : independentFilters(element=element,filterIndex=x))(channels)
            lstm=Bidirectional(LSTM(50,return_sequences=True),merge_mode='concat')(filterImg)
            lstmsVec.append(lstm)
            #lstm=LSTM(50,stateful=True)(filterImg) #batch_shape=(512, 7, 5)
            #denselayers=Dense(400)(lstm)
            #denselayers=Activation("relu")(denselayers)
            #denselayers=Dropout(0.5)(denselayers)
            #denselayers=Dense(150)(denselayers)
            #denselayers=Activation("relu")(denselayers)
            #denselayers=Dropout(0.8)(denselayers)
            channels=Flatten()(lstm)
            classificationLayer=Dense(classes)(channels)
            classificationLayer=Activation("softmax")(classificationLayer)
            
            lstmsVec.append(classificationLayer)
        #base example for 2 lstms     
        #x0 = Lambda(lambda x : x[:,0,:,:])(out1)
        #x1 = Lambda(lambda x : x[:,1,:,:])(out1)
        #lstm1=LSTM(40,input_shape=(5,5))(x0)
        #lstm2=LSTM(40,input_shape=(5,5))(x1)
        
        #merged = merge(lstmsVec, mode='concat',concat_axis=1)
        
        
        # a softmax classifier
        #classificationLayer=Dense(classes)(merged)
        #channels=Flatten()(channels)
        #classificationLayer=Dense(classes)(channels)
        #classificationLayer=Activation("softmax")(classificationLayer)
        
        model=Model(inputNet,classificationLayer)
        #model=Model(inputNet,lstmsVec)
        return model


class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        # CONV => RELU => POOL
        # kernel size (width, height) default (2,5)
        model.add(Conv2D(20, kernel_size=(2,5), padding="same",
            input_shape=input_shape))
        model.add(Activation("relu"))
        # pool size - down scale int factor (vertical, horizontal)
        model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 1)))
        # CONV => RELU => POOL
        
        model.add(Conv2D(50, kernel_size=(3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 1))) #1,2
        model.summary()
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
       
#------------- load Data ----------------

# special cross validation file 1 NNNormalizeData-out-set-0.csv
#df = pd.read_csv('../NNNormalizeData-out-set-1.csv',header=None)
#df = pd.read_csv('../NNNormalizeData-out.csv',header=None)
df = pd.read_csv('../NNNormalizeData-out.csv',header=None)  # 3 classes : up neutral down

np.random.seed(42) # always shuffle the same way 
df = df.reindex(np.random.permutation(df.index)) # shuffle examples 
df.reset_index(inplace=True, drop=True)

print(df)

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
print('***************************')

print(inputs)

total_inputs,total_output = df.as_matrix(inputs).astype(np.float32),df.as_matrix([target]).astype(np.int32)
print('###########################')
#print(total_inputs)
print(total_inputs.shape)

total_inputs = np.reshape(total_inputs, (-1,5,7))
print(total_inputs.shape)
print (type(total_inputs))
print(total_inputs[0])
#total= np.array([][][])
#total.append(total_inputs[0].T)
#total.append(total_inputs[1].T)
total=np.array([total_inputs[0].T])
for i in total_inputs:
	total=np.row_stack((total, [i.T]))
total=np.delete(total, 0, 0)
#total=np.row_stack((total, [total_inputs[2].T]))
#np.concatenate(total, total_inputs[1].T)
print(total.shape)
print (type(total))
print (total[0])

#total_inputs[0]=total_inputs[0].T
#print(total)
print('---------------------------')
#print(total_inputs)
#print(total_inputs.shape)

X_train=total#_inputs
y_train=total_output
print ("ytrain   ### " , y_train)

#X_traiINPUT_SHAPEn, y_train , X_test, y_test
#X_train, X_test, y_train , y_test = train_test_split(total_inputs, total_output, test_size=0.15, random_state=42)

#-------------- End Load Data -----------





NB_EPOCH = 500
# network and training
BATCH_SIZE = 20
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT=0.2

IMG_ROWS, IMG_COLS = 7, 5 # input image dimensions
NB_CLASSES = 5  # number of outputs = number of classes
INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)


K.set_image_dim_ordering("th")

# consider them as float and normalize
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
print("-------------- [0 , 1] ----------------------")
X_train += 1 
X_train /= 2
print(X_train)
#X_test += 1  

# we need a 60K x [1 x 28 x 28] shape as input to the CONVNET

############## comment this line to use LSTM uncomment to use CNN
X_train = X_train[:, np.newaxis, :, :]


#X_test = X_test[:, np.newaxis, :, :]

print(X_train.shape, 'train samples')
#print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
#y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# initialize the optimizer and model
model = RJPGNet.build(timeSteps=7,variables=5,classes=5,nlstms=1)
#model = LSTMNet.build(timeSteps=7,variables=5,classes=5)
#model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)

model.summary()
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
	metrics=["accuracy"])

plot_model(model, to_file="model2.png",show_shapes=True)

# Prepare saver.
builder = tf.saved_model.builder.SavedModelBuilder("./model_keras")

# Initialize all variables
sess.run(tf.global_variables_initializer())


tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
esCallBack = EarlyStopping(monitor='val_acc', min_delta=0, patience=12, verbose=0, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)

print ("ytrain   ### " , y_train)

history = model.fit(X_train, y_train,# ,y_train,y_train], 
		batch_size=BATCH_SIZE, epochs=NB_EPOCH, 
		verbose=1, # 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
		validation_split=VALIDATION_SPLIT, callbacks=[tbCallBack,reduce_lr])#,esCallBack])

# Save model so we can use it in java.
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
builder.save(True)

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
#writer = tf.summary.FileWriter('./keras_board/1')
#writer.add_graph(sess.graph)


######################## LOAD TEST DATA ###################################
df_test = pd.read_csv('../NNNormalizeData-out-test.csv',header=None)

np.random.seed(42) # always shuffle the same way 
df_test = df_test.reindex(np.random.permutation(df_test.index)) # shuffle examples 
df_test.reset_index(inplace=True, drop=True)

inputs_test = []
target_test = []

y=0;    
for x in df_test.columns:
    if y != 35 :
        #print("added %d" %y)
        inputs_test.append(x)
    else :
        target_test.append(x)
    y+=1



X_test, y_test = df_test.as_matrix(inputs_test).astype(np.float32),df_test.as_matrix([target_test]).astype(np.int32)

X_test = np.reshape(X_test, (-1,5,7))

X_test = X_test[:, np.newaxis, :, :]
y_test = np_utils.to_categorical(y_test, NB_CLASSES)


score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])

# list all data in history
print(history.history.keys())
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

sample_1 = np.array( [[[0.37671986791414125,0.28395908337619136,-0.0966095873607713,-1.0,0.06891621389763203,-0.09716678086712205,0.726029084013637],
					[4.984689881073479E-4,-0.30296253267499107,-0.16192917054985334,0.04820256230479658,0.4951319883569152,0.5269983894210499,-0.2560313828048315],
					[-0.3710980821053321,-0.4845867212612598,-0.8647234314469595,-0.6491591208322198,-1.0,-0.5004549422844073,-0.9880910165770813],
					[0.5540293108747256,0.5625990251930839,0.7420121698556554,0.5445551415657979,0.4644276850235627,0.7316976292340245,0.636690006814346],
					[0.16486621649984112,-0.0466018967678159,0.5261100063227044,0.6256168612312738,-0.544295484930702,0.379125782517193,0.6959368575211544]]], dtype=float)
sample_1 = sample_1[:, np.newaxis, :, :]
print(sample_1)
out = model.predict_proba(sample_1, batch_size=1, verbose=1)
print(out)

out = model.predict(sample_1, batch_size=1, verbose=1)
print(out)

out = model.predict_classes(sample_1, batch_size=1, verbose=1)
print(out)
