import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

seed=120
a=np.arange(1,10,1)
b=np.arange(1,10,1)
np.random.seed(seed)
np.random.shuffle(a)
np.random.seed(seed)
np.random.shuffle(b)
c=np.arange(1,10,1)
k=tf.convert_to_tensor(c)
k=tf.random_shuffle(k,seed=seed)
sess=tf.Session()
print(a,b,sess.run(k))