'''
Created on 21/12/2017

@author: rjpg
'''

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.backend.tensorflow_backend import reshape

input_img = Input(shape=(28, 28,1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

encoded = MaxPooling2D((2, 2), padding='same',name='max_encode')(x)
y = reshape(encoded, [4,4])
print("-------------------------")
print((y))
print("-------------------------")
# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.summary()


testSample=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]
testSample=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]
inp= Input(shape=(28, 28, 1))
print(inp)