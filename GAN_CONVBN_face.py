import numpy as np
import  h5py
from skimage.io import imsave
import struct
import matplotlib.pyplot as plt
import shutil
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


file = h5py.File('emo4424.h5','r')
train_X = file['Input'][:]
print(train_X.shape[0])
def getBatch(Batchsize):
    Input=[]
    for i in range(Batchsize):
        index = np.random.randint(0, train_X.shape[0])
        Input.append(train_X[index,:])
    return np.array(Input)

img_height = 64
img_width = 64
def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    batch_res = batch_res.reshape((batch_res.shape[0], img_height, img_width,3))
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w,3), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res+1)/2 * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w,:] = img
    imsave(fname, img_grid)

def getNoise(Batchsize,length):
    return np.random.normal(0, 1,size=[Batchsize,length])

def Leakyrelu(x, alpha=0.2, max_value=None):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x

def batch_norm(inputs, is_training,is_conv_out=True,decay = 0.999):
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
  initial = tf.truncated_normal(shape, stddev=0.02)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial)

#Param
Ktrain=1
Nsize=16
Dsize=16
NoiseP=tf.placeholder(tf.float32,shape=[None,100])
TpX=tf.placeholder(tf.float32,shape=[None,64,64,3])
# NoisePX=tf.reshape(NoiseP,shape=[-1,4,4,6])
# TpX=tf.reshape(Tp,shape=[-1,28,28,1])

#Generator

WGfc1=weight_variable([100,16*1024])
BGfc1=bias_variable([16*1024])
NoisePX=tf.nn.relu(batch_norm(tf.matmul(NoiseP,WGfc1)+BGfc1,BNistrain,False))
NoisePX=tf.reshape(NoisePX,shape=[-1,4,4,1024])

Wdconv1=weight_variable([5,5,512,1024])
Bdconv1=bias_variable([512])
dconv1=tf.nn.relu(batch_norm(tf.nn.conv2d_transpose(NoisePX,Wdconv1,output_shape=[Nsize,8,8,512],strides=[1,2,2,1],padding='SAME')+Bdconv1,BNistrain))

Wdconv2=weight_variable([5,5,256,512])
Bdconv2=bias_variable([256])
dconv2=tf.nn.relu(batch_norm(tf.nn.conv2d_transpose(dconv1,Wdconv2,output_shape=[Nsize,16,16,256],strides=[1,2,2,1],padding='SAME')+Bdconv2,BNistrain))

Wdconv3=weight_variable([5,5,128,256])
Bdconv3=bias_variable([128])
dconv3=tf.nn.relu(batch_norm(tf.nn.conv2d_transpose(dconv2,Wdconv3,output_shape=[Nsize,32,32,128],strides=[1,2,2,1],padding='SAME')+Bdconv3,BNistrain))

Wdconv4=weight_variable([5,5,3,128])
Bdconv4=bias_variable([3])
GenOUT=tf.nn.tanh(tf.nn.conv2d_transpose(dconv3,Wdconv4,output_shape=[Nsize,64,64,3],strides=[1,2,2,1],padding='SAME')+Bdconv4)
g_params = [Wdconv1,Bdconv1,Wdconv2,Bdconv2,Wdconv3,Bdconv3,Wdconv4,Bdconv4,WGfc1,BGfc1]

BNtrain=True
#Discriminator
Wconv1 = weight_variable([3, 3, 3, 64])
Bconv1 = bias_variable([64])
GConv1 = Leakyrelu(tf.nn.conv2d(TpX, Wconv1, strides=[1, 1, 1, 1], padding='SAME') + Bconv1)
DConv1 = Leakyrelu(tf.nn.conv2d(GenOUT, Wconv1, strides=[1, 1, 1, 1], padding='SAME') + Bconv1)

Wconv2 = weight_variable([3, 3, 64, 64])
Bconv2 = bias_variable([64])
GConv2 = Leakyrelu(batch_norm(tf.nn.conv2d(GConv1, Wconv2, strides=[1, 2, 2, 1], padding='SAME') + Bconv2, BNtrain))
DConv2 = Leakyrelu(batch_norm(tf.nn.conv2d(DConv1, Wconv2, strides=[1, 2, 2, 1], padding='SAME') + Bconv2, BNtrain))

Wconv3 = weight_variable([3, 3, 64, 128])
Bconv3 = bias_variable([128])
GConv3 = Leakyrelu(batch_norm(tf.nn.conv2d(GConv2, Wconv3, strides=[1, 1, 1, 1], padding='SAME') + Bconv3, BNtrain))
DConv3 = Leakyrelu(batch_norm(tf.nn.conv2d(DConv2, Wconv3, strides=[1, 1, 1, 1], padding='SAME') + Bconv3, BNtrain))

Wconv4 = weight_variable([3, 3, 128, 128])
Bconv4 = bias_variable([128])
GConv4 = Leakyrelu(batch_norm(tf.nn.conv2d(GConv3, Wconv4, strides=[1, 2, 2, 1], padding='SAME') + Bconv4, BNtrain))
DConv4 = Leakyrelu(batch_norm(tf.nn.conv2d(DConv3, Wconv4, strides=[1, 2, 2, 1], padding='SAME') + Bconv4, BNtrain))

Wconv5 = weight_variable([3, 3, 128, 256])
Bconv5 = bias_variable([256])
GConv5 = Leakyrelu(batch_norm(tf.nn.conv2d(GConv4, Wconv5, strides=[1, 1, 1, 1], padding='SAME') + Bconv5, BNtrain))
DConv5 = Leakyrelu(batch_norm(tf.nn.conv2d(DConv4, Wconv5, strides=[1, 1, 1, 1], padding='SAME') + Bconv5, BNtrain))

Wconv6 = weight_variable([3, 3, 256, 256])
Bconv6 = bias_variable([256])
GConv6 = Leakyrelu(batch_norm(tf.nn.conv2d(GConv5, Wconv6, strides=[1, 2, 2, 1], padding='SAME') + Bconv6, BNtrain))
DConv6 = Leakyrelu(batch_norm(tf.nn.conv2d(DConv5, Wconv6, strides=[1, 2, 2, 1], padding='SAME') + Bconv6, BNtrain))

Wconv7 = weight_variable([3, 3, 256, 512])
Bconv7 = bias_variable([512])
GConv7 = Leakyrelu(batch_norm(tf.nn.conv2d(GConv6, Wconv7, strides=[1, 1, 1, 1], padding='SAME') + Bconv7, BNtrain))
DConv7 = Leakyrelu(batch_norm(tf.nn.conv2d(DConv6, Wconv7, strides=[1, 1, 1, 1], padding='SAME') + Bconv7, BNtrain))

Wconv8 = weight_variable([3, 3, 512, 512])
Bconv8 = bias_variable([512])
GConv8 = Leakyrelu(batch_norm(tf.nn.conv2d(GConv7, Wconv8, strides=[1, 2, 2, 1], padding='SAME') + Bconv8, BNtrain))
DConv8 = Leakyrelu(batch_norm(tf.nn.conv2d(DConv7, Wconv8, strides=[1, 2, 2, 1], padding='SAME') + Bconv8, BNtrain))
GConv8_line = tf.reshape(GConv8, shape=[-1, 2048])
DConv8_line = tf.reshape(DConv8, shape=[-1, 2048])

Wfc1 = weight_variable([2048, 1024])
Bfc1 = bias_variable([1024])
Gfc1 = Leakyrelu(tf.matmul(GConv8_line, Wfc1) + Bfc1)
Dfc1 = Leakyrelu(tf.matmul(DConv8_line, Wfc1) + Bfc1)
# Dfc1 = tf.nn.dropout(Dfc1, 0.7)
# Gfc1 = tf.nn.dropout(Gfc1, 0.7)

Wfc2 = weight_variable([1024, 1])
Bfc2 = bias_variable([1])
D1 = tf.nn.sigmoid(tf.matmul(Gfc1, Wfc2) + Bfc2)
D2 = tf.nn.sigmoid(tf.matmul(Dfc1, Wfc2) + Bfc2)

d_params = [Wconv1, Bconv1, Wconv2, Bconv2, Wconv3, Bconv3, Wconv4, Bconv4, Wconv5, Bconv5, Wconv6, Bconv6, Wconv7,
           Bconv7, Wconv8, Bconv8, Wfc1, Bfc1, Wfc2, Bfc2]
# Wconv1=weight_variable([5,5,3,64])
# Bconv1=bias_variable([64])
# D1conv1=Leakyrelu(tf.nn.conv2d(TpX,Wconv1,strides=[1,2,2,1],padding='SAME')+Bconv1)
# D2conv1=Leakyrelu(tf.nn.conv2d(GenOUT,Wconv1,strides=[1,2,2,1],padding='SAME')+Bconv1)
#
# Wconv2=weight_variable([5,5,64,128])
# Bconv2=bias_variable([128])
# D1conv2=Leakyrelu(batch_norm(tf.nn.conv2d(D1conv1,Wconv2,strides=[1,2,2,1],padding='SAME')+Bconv2,BNistrain))
# D2conv2=Leakyrelu(batch_norm(tf.nn.conv2d(D2conv1,Wconv2,strides=[1,2,2,1],padding='SAME')+Bconv2,BNistrain))
#
# Wconv3=weight_variable([5,5,128,64])
# Bconv3=bias_variable([64])
# D1conv3=Leakyrelu(batch_norm(tf.nn.conv2d(D1conv2,Wconv3,strides=[1,2,2,1],padding='SAME')+Bconv3,BNistrain))
# D2conv3=Leakyrelu(batch_norm(tf.nn.conv2d(D2conv2,Wconv3,strides=[1,2,2,1],padding='SAME')+Bconv3,BNistrain))
#
# Wconv4=weight_variable([5,5,64,32])
# Bconv4=bias_variable([32])
# D1conv4=Leakyrelu(batch_norm(tf.nn.conv2d(D1conv3,Wconv4,strides=[1,2,2,1],padding='SAME')+Bconv4,BNistrain))
# D2conv4=Leakyrelu(batch_norm(tf.nn.conv2d(D2conv3,Wconv4,strides=[1,2,2,1],padding='SAME')+Bconv4,BNistrain))
# # print(D2conv4.shape)
#
# D1conv4_line=tf.reshape(D1conv4,shape=[-1,4*4*32])
# D2conv4_line=tf.reshape(D2conv4,shape=[-1,4*4*32])
#
# Wfc1=weight_variable([4*4*32,64])
# Bfc1=bias_variable([64])
# D1fc1=Leakyrelu(tf.matmul(D1conv4_line,Wfc1)+Bfc1)
# D2fc1=Leakyrelu(tf.matmul(D2conv4_line,Wfc1)+Bfc1)
#
# Wfc2=weight_variable([64,1])
# Bfc2=bias_variable([1])
# D1=tf.nn.sigmoid(tf.matmul(D1fc1,Wfc2)+Bfc2)
# D2=tf.nn.sigmoid(tf.matmul(D2fc1,Wfc2)+Bfc2)
# d_params =[Wconv1,Bconv1,Wconv2,Bconv2,Wconv3,Bconv3,Wconv4,Bconv4,Wfc1,Bfc1,Wfc2,Bfc2]
# Wconv5=weight_variable([4,4,32,1])
# Bconv5=bias_variable([1])
# D1conv5=tf.nn.sigmoid(tf.nn.conv2d(D1conv4,Wconv5,strides=[1,1,1,1],padding='VALID')+Bconv5)
# D2conv5=tf.nn.sigmoid(tf.nn.conv2d(D2conv4,Wconv5,strides=[1,1,1,1],padding='VALID')+Bconv5)

# D1=tf.reshape(D1conv5,shape=[-1,1])
# D2=tf.reshape(D2conv5,shape=[-1,1])
# d_params =[Wconv1,Bconv1,Wconv2,Bconv2,Wconv3,Bconv3,Wconv4,Bconv4,Wconv5,Bconv5]

loss1=-tf.reduce_sum(tf.log(tf.clip_by_value(D1,1e-10,1.0))+tf.log(tf.clip_by_value(1-D2,1e-10,1.0)))
TrainStep1=tf.train.AdamOptimizer(0.0001).minimize(loss1,var_list=d_params)
loss2=-tf.reduce_sum(tf.log(tf.clip_by_value(D2,1e-10,1.0)))
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
        result1,err1=sess.run([TrainStep1,loss1],feed_dict={NoiseP:Noise_batch,TpX:Pdata_batch})
    Pdata_batch = getBatch(Dsize)
    # Pdata_batch, _ = mnist.train.next_batch(Dsize)srgan

    Noise_batch = getNoise(Nsize,100)
    GEN,result2,err2=sess.run([GenOUT,TrainStep2,loss2],feed_dict={NoiseP:Noise_batch,TpX:Pdata_batch})
    # GEN,err2=sess.run([GenOUT,loss2],feed_dict={NoiseP:Noise_batch,TpX:Pdata_batch})

    print('Dloss:',err1,'Gloss:',err2)
    if (iter+1)%100==0:
        show_result(GEN, "output/sample{0}.jpg".format(iter))
    if (iter+1)%3000==0:
        path = saver.save(sess,'GAN_session/session_GANmnistx.ckpt')
        print(path)