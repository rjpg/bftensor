'''
Created on 29/12/2017

@author: rjpg
'''

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.layers.pooling import AveragePooling2D
from numpy import vstack, hstack
from keras.legacy.layers import merge
from keras.layers.core import Lambda

K.set_image_dim_ordering("th")

input_img = Input(shape=(1, 28, 28))  # adapt this if using `channels_first` image data format

x = Conv2D(5, (3, 3), activation='relu', padding='same')(input_img)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = AveragePooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid')(x)
lstmsVec=[]
for x in range(0,5):
    filterImg=Lambda(lambda element : element[:,x,:,:])(encoded)
    lstmsVec.append(filterImg)
merged = merge(lstmsVec, mode='concat',concat_axis=2)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(5, (3, 3), activation='relu', padding='same')(encoded)
#x = UpSampling2D((2, 2))(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
#x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 1))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

encoder = Model(input_img, encoded)

encoderHstack= Model(input_img,merged)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.summary()




from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))  # adapt this if using `channels_first` image data format


from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# do not train encoding 
for layer in encoderHstack.layers : layer.trainable = False

#inputNet = Input(shape=(7,10),batch_shape=(20, 7, 5)) 
        #lstm=Bidirectional(LSTM(100,recurrent_dropout=0.4,dropout=0.4),merge_mode='concat')(inputNet) #worse using stateful=True
#lstm=Bidirectional(LSTM(50),merge_mode='concat')(inputNet) #worse using stateful=True 
        #denselayers=Dense(400)(lstm)
        #denselayers=Activation("relu")(denselayers)
        #denselayers=Dropout(0.5)(denselayers)
        #denselayers=Dense(150)(denselayers)
        #denselayers=Activation("relu")(denselayers)
        #denselayers=Dropout(0.8)(denselayers)
        # a softmax classifier
#classificationLayer=Dense(classes,activation='softmax')(lstm)
#classificationLayer=Activation("softmax")(classificationLayer)
        
#model=Model(inputNet,classificationLayer)

decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
#n = 10
#plt.figure(figsize=(20, 4))
#for i in range(n):
#    # display original
#    ax = plt.subplot(2, n, i+1)
#    plt.imshow(x_test[i].reshape(28, 28))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)

#    # display reconstruction
#    ax = plt.subplot(2, n, (i+1) + n)
#    plt.imshow(decoded_imgs[i].reshape(28, 28))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#plt.show()




encoded_imgs = encoder.predict(x_test)


#array=[]
#nimages=encoded_imgs[0,0,0,:].size
#hight=encoded_imgs[0,0,:,0].size
#wigth=encoded_imgs[0,:,0,0].size

#for i in range(hight):
#    array.append([])
#for i in range(nimages):
#    for x in range(hight):
#        array[x]=sum(array[x],encoded_imgs[0,:,x,i])

#print(array.shape)

#print (array)
images = encoded_imgs[0,:,:,:]

print(images[:,1,1].size)

image=images[0,:,:]
for i in range(1,5):
    image=hstack((image,images[i,:,:]))


plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 1, 1)
plt.imshow(image)
#plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()


print("showing encoded Hstack with merge filters")
encoded_imgsHstak = encoderHstack.predict(x_test)
print("showing encoded Hstack with merge filters")
images = encoded_imgsHstak[0,:,:]

plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 1, 1)
plt.imshow(image)
#plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()



n = 5
plt.figure(figsize=(40, 5))

for i in range(n):
    # display original
#    ax = plt.subplot(2, 10, i+1)
#    plt.imshow(x_test[0].reshape(28, 28))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)


    ax = plt.subplot(2, 10, (i+1)) #+ n
    #plt.imshow(encoded_imgs[0,:,:,i])#.reshape(14, 14))
    plt.imshow(encoded_imgs[0,i,:,:])
#    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    #ax = plt.subplot(4, 10, (i+1)+n)
    #plt.imshow(x_test[1].reshape(28, 28))
    #plt.gray()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)

    
plt.show()

n = 5
plt.figure(figsize=(40, 5))


for i in range(n):
    # display original
    ax = plt.subplot(2, 10, i+1)
    plt.imshow(x_test[1].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    ax = plt.subplot(2, 10, (i+1) + n)
    plt.imshow(encoded_imgs[1,i,:,:])#.reshape(14, 14))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    #ax = plt.subplot(4, 10, (i+1)+n)
    #plt.imshow(x_test[1].reshape(28, 28))
    #plt.gray()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)

    
plt.show()
