import numpy as np
from skimage.io import imsave
import struct
import matplotlib.pyplot as plt
from readmnist import DataUtils
import shutil
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

trainfile_X = 'MNIST/train-images.idx3-ubyte'
trainfile_y = 'MNIST/train-labels.idx1-ubyte'
testfile_X = 'MNIST/t10k-images.idx3-ubyte'
testfile_y = 'MNIST/t10k-labels.idx1-ubyte'
train_X = DataUtils(filename=trainfile_X).getImage()
train_y = DataUtils(filename=trainfile_y).getLabel()
test_X = DataUtils(testfile_X).getImage()
test_y = DataUtils(testfile_y).getLabel()
train_X=2*train_X-1

def getBatch(Batchsize):
    Input=[]
    for i in range(Batchsize):
        index = np.random.randint(0, train_X.shape[0])
        # plt.imshow(np.reshape(train_X[index,:],[28,28]))
        # plt.show()
        Input.append(train_X[index,:])
    return np.array(Input)

img_height = 28
img_width = 28
def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    batch_res = batch_res.reshape((batch_res.shape[0], img_height, img_width))
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)

def getNoise(Batchsize,length):
    return np.random.normal(0, 1,size=[Batchsize,length])


def batch_norm(inputs, is_training,is_conv_out=False,decay = 0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, 0.001)

BNistrain=True

#Network Strat
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial)

#Param
Ktrain=1
Nsize=256
Dsize=256
NoiseP=tf.placeholder(tf.float32,shape=[None,100])
Tp=tf.placeholder(tf.float32,shape=[None,784])


#Generator
WGFc1=weight_variable([100,150])
BGFc1=bias_variable([150])
GFc1=tf.nn.relu(tf.matmul(NoiseP,WGFc1)+BGFc1)

WGFc2=weight_variable([150,300])
BGFc2=bias_variable([300])
GFc2=tf.nn.relu(tf.matmul(GFc1,WGFc2)+BGFc2)

WGFc3=weight_variable([300,784])
BGFc3=bias_variable([784])
GFc3=tf.nn.tanh(tf.matmul(GFc2,WGFc3)+BGFc3)
GenOUT=tf.reshape(GFc3,shape=[-1,28,28])
g_params = [WGFc1, BGFc1, WGFc2, BGFc2, WGFc3, BGFc3]

#Discriminator
# Xp=tf.concat([Tp,GFc3],axis=0)
WDFc1=weight_variable([784,300])
BDFc1=bias_variable([300])
D1Fc1= tf.nn.dropout(tf.nn.relu(tf.matmul(Tp,WDFc1)+BDFc1),0.7)
D2Fc1= tf.nn.dropout(tf.nn.relu(tf.matmul(GFc3,WDFc1)+BDFc1),0.7)

WDFc2=weight_variable([300,150])
BDFc2=bias_variable([150])
D1Fc2=tf.nn.dropout(tf.nn.relu(tf.matmul(D1Fc1,WDFc2)+BDFc2),0.7)
D2Fc2=tf.nn.dropout(tf.nn.relu(tf.matmul(D2Fc1,WDFc2)+BDFc2),0.7)

WDFc3=weight_variable([150,1])
BDFc3=bias_variable([1])
D1=tf.nn.sigmoid(tf.matmul(D1Fc2,WDFc3)+BDFc3)
D2=tf.nn.sigmoid(tf.matmul(D2Fc2,WDFc3)+BDFc3)
d_params = [WDFc1, BDFc1, WDFc2, BDFc2, WDFc3, BDFc3]

# D1=tf.slice(OUT, [0, 0], [Dsize, -1], name=None)
# D2=tf.slice(OUT, [Dsize, 0], [-1, -1], name=None)
loss1=-tf.reduce_mean(tf.log(tf.clip_by_value(D1,1e-10,1.0))+tf.log(tf.clip_by_value(1-D2,1e-10,1.0)))
TrainStep1=tf.train.AdamOptimizer(0.0001).minimize(loss1,var_list=d_params)
loss2=-tf.reduce_mean(tf.log(tf.clip_by_value(D2,1e-10,1.0)))
TrainStep2=tf.train.AdamOptimizer(0.0001).minimize(loss2,var_list=g_params)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
init=tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()
# saver.restore(sess,'GAN_session/session_GANmnist.ckpt')

# for i in range(10):
#     Noise_batch = getNoise(Nsize, 100)
#     Pdata_batch = getBatch(Dsize)
#     Gtrain = False
#     Dtrain = True
#     result1, err1 = sess.run([TrainStep1, loss1], feed_dict={NoiseP: Noise_batch, Tp: Pdata_batch})
output_path='output'
# os.rmdir(output_path)
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.mkdir(output_path)
for iter in range(1000000):
    print('\n',iter)
    for i in range(Ktrain):
        Noise_batch=getNoise(Nsize,100)
        Pdata_batch=getBatch(Dsize)
        # Pdata_batch, _ = mnist.train.next_batch(Dsize)
        result1,err1=sess.run([TrainStep1,loss1],feed_dict={NoiseP:Noise_batch,Tp:Pdata_batch})
    Pdata_batch = getBatch(Dsize)
    # Pdata_batch, _ = mnist.train.next_batch(Dsize)
    Noise_batch = getNoise(Nsize,100)
    GEN,result2,err2=sess.run([GenOUT,TrainStep2,loss2],feed_dict={NoiseP:Noise_batch,Tp:Pdata_batch})

    print('Dloss:',err1,'Gloss:',err2)
    if (iter+1)%100==0:
        show_result(GEN, "output/sample{0}.jpg".format(iter))
    if (iter+1)%3000==0:
        path = saver.save(sess,'GAN_session/session_GANmnist.ckpt')
        print(path)