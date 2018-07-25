'''
Created on 23/07/2018

@author: rjpg
'''
import numpy as np

def transform(element,n):
    batchResultFlat=[]
    for batchElement in element:
        resultFlat=[]
        for cnnFilterImage in batchElement:
            firstColumns=cnnFilterImage[:,list(range(n))]
            result=np.concatenate((cnnFilterImage,np.array(firstColumns)),axis=1)
            
            lastColumnsReverse=cnnFilterImage[:,list(range(cnnFilterImage.shape[1]-n,cnnFilterImage.shape[1]))]
            result=np.concatenate((np.array(lastColumnsReverse),result),axis=1)
            resultFlat.append(result)
        batchResultFlat.append(np.array(resultFlat))
    newElement = np.array(batchResultFlat)
    return newElement

import tensorflow as tf
def transform2(elementA, n=2):
    x_unpacked = tf.unstack(elementA)#Xtensor)
    batchResultFlat=[]
    for batchElement in x_unpacked:
        resultFlat=[]
        batchElementUnpack = tf.unstack(batchElement)
        for cnnFilterImage in batchElementUnpack:
            
            ind = tf.constant(list(range(n)))
            firstColumns = tf.transpose(tf.nn.embedding_lookup(tf.transpose(cnnFilterImage), ind))
            
            result = tf.concat([cnnFilterImage,firstColumns], axis=1)
            
            ind = tf.constant(list(range(cnnFilterImage.shape[1]-n,cnnFilterImage.shape[1])))
            lastColumns = tf.transpose(tf.nn.embedding_lookup(tf.transpose(cnnFilterImage), ind))
            
            result = tf.concat([lastColumns,result], axis=1)
 
            resultFlat.append(result)
 
        batchResultFlat.append(tf.stack(resultFlat))
    ret = tf.stack(batchResultFlat)
    return ret

def transform3(elementA, n=2):
    firstColumns=testElement[:,:,:,list(range(n))]
    lastColumns=testElement[:,:,:,list(range(testElement.shape[3]-n,testElement.shape[3]))]
    result=np.concatenate((testElement,firstColumns),axis=3)
    result=np.concatenate((lastColumns,result),axis=3)
    return result

def transform4(elementA, n=2):
    firstColumns=testElement[:,:,:,0:n]
    lastColumns=testElement[:,:,:,testElement.shape[3]-n:testElement.shape[3]]
    result=tf.concat([testElement,firstColumns], axis=3)
    #result=np.concatenate((testElement,firstColumns),axis=3)
    result=tf.concat([lastColumns,result], axis=3)
    #result=np.concatenate((lastColumns,result),axis=3)
    return result



### Testing ###
testElement= np.array([[[[1.,2,3],[4,5,6],[7,8,9],[10,11,12]],
                        [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]],
                        
                        [[[10,20,30],[40,50,60],[70,80,90],[100,110,120]],
                        [[10,20,30],[40,50,60],[70,80,90],[100,110,120]]] ])

testElement[:,:,]


print(testElement.shape)
print(testElement)

print("--calling transform--")
testElementTranformed = transform4(testElement, n=2)
print("-- End calling transform--")

# print tensor to confirm
sess = tf.InteractiveSession()
c = sess.run(testElementTranformed)
print(c)
sess.close()



print (testElementTranformed)
print (testElementTranformed.shape)