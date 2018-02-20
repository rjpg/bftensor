'''
Created on 19/12/2017

@author: rjpg
'''


from keras.datasets import mnist 
from keras.models import Model 
from keras.layers import Input, Dense 
from keras.utils import np_utils 
import numpy as np
from tensorflow.python.ops.variables import trainable_variables

num_train = 60000
num_test = 10000

height, width, depth = 28, 28, 1 # MNIST images are 28x28
num_classes = 10 # there are 10 classes (1 per digit)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(num_train, height * width)
X_test = X_test.reshape(num_test, height * width)
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')

X_train /= 255 # Normalise data to [0, 1] range
X_test /= 255 # Normalise data to [0, 1] range

Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

input_img = Input(shape=(height * width,))

x = Dense(height * width, activation='relu')(input_img)

encoded1 = Dense(height * width//2, activation='relu')(x)
encoded2 = Dense(height * width//8, activation='relu')(encoded1)

y = Dense(height * width//256, activation='relu')(encoded2)

decoded2 = Dense(height * width//8, activation='relu')(y)
decoded1 = Dense(height * width//2, activation='relu')(decoded2)

z = Dense(height * width, activation='sigmoid')(decoded1)
autoencoder = Model(input_img, z)

#encoder is the model of the autoencoder slice in the middle 
encoder = Model(input_img, y)

autoencoder.compile(optimizer='adadelta', loss='mse') # reporting the loss

autoencoder.fit(X_train, X_train,
      epochs=3,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test, X_test))

# if you want an encoded flatten representation of every test MNIST
reduced_representation =encoder.predict(X_test)

#print encoded1 weights
#weights = autoencoder.layers[1].get_weights() # list of numpy arrays
#print(weights)

# if you want to lock the weights of the encoder on post-training 
#for layer in encoder.layers : layer.trainable = False


# define new model encoder->Dense  10 neurons with soft max for classification 
out2 = Dense(num_classes, activation='softmax')(encoder.output)
newmodel = Model(encoder.input,out2)


newmodel.compile(loss='categorical_crossentropy',
          optimizer='adam', 
          metrics=['accuracy']) 



newmodel.fit(X_train, Y_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test, Y_test))

#print encoded1 weights again 
#weights = newmodel.layers[1].get_weights() # list of numpy arrays
#print(weights)


scores = newmodel.evaluate(X_test, Y_test, verbose=1) 
print("Accuracy: ", scores[1])
 
 