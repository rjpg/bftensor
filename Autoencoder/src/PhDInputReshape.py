'''
Created on 10/06/2017

@author: rjpg
'''

import numpy as np
import pandas as pd

df = pd.read_csv('../NNNormalizeData-out.csv',header=None)

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

print(inputs)

total_inputs,total_output = df.as_matrix(inputs).astype(np.float32),df.as_matrix([target]).astype(np.int32)

print(total_inputs)

total_inputs = np.reshape(total_inputs, (-1,5,7))
print('---------------------------')
print(total_inputs)