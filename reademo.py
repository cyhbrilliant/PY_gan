# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import h5py

Input =[]
for i in range(4424):
    print(i)
    tempimg = cv2.imread('webemo_ts/'+str(i)+'.jpg')
    if tempimg.shape[0]<64:
        continue
    tempimg=cv2.resize(tempimg,(64,64))
    tempimg=cv2.cvtColor(tempimg,cv2.COLOR_BGR2RGB)
    tempimg=(tempimg/255.)*2.-1.
    # plt.imshow(tempimg)
    # plt.show()
    Input.append(tempimg)
Input=np.array(Input)
print(Input.shape)
print('OK')
file = h5py.File('emo4424.h5','w')
file.create_dataset('Input', data = Input)