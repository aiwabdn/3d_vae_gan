from __future__ import with_statement
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from lib.read import ShapeNetLoader
from lib.voxel import voxel2obj
import time
from os import system

local = False

#if local:
#    DATA_DIR = '/Users/anindya/Downloads/'
#    OUT_DIR = '/Users/anindya/Downloads/output/'
#    BATCH_SIZE = 1
#    DISC_ITERS = 1
#    EPOCHS = 3 
#else:
DATA_DIR = 'gs://shapenet_data'
OUT_DIR = '/tmp' #'gs://3dwgan'
BATCH_SIZE = 50
BATCHES_PER_EPOCH = 9 
DISC_ITERS = 5
EPOCHS = 701


INPUT_DIMS = [200, 200, 1]
ENCODING_DIM = 200
OUTPUT_DIMS = [32, 32, 32]

# gradient penalty multiplier for WGAN
LAMBDA = 10
EPSILON = 1e-3
LEARNING_RATE = 1e-4
BETA_1=0.5
BETA_2=0.9

# loss weights from 3DGAN paper
ALPHA_1 = 5
ALPHA_2 = 0.001

# input placeholder
input_img = tf.placeholder(tf.float32, [None]+INPUT_DIMS, name='input_img_placeholder')
input_volume = tf.placeholder(tf.float32, [None]+OUTPUT_DIMS, name='input_volume_placeholder')
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

# encoding prior to draw samples from
prior_mean = tf.zeros([ENCODING_DIM], name='prior_mean')
prior_var = tf.ones([ENCODING_DIM], name='prior_var')
prior = tf.contrib.distributions.MultivariateNormalDiag(prior_mean, prior_var, name='prior_dist')

def leaky_relu(x, alpha=0.2, max_value=None):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    x -= alpha * negative_part
    return x

def conv2d_block(inputs, out_channels, kernel_size=2, stride=1, activation=tf.nn.relu, batch_norm=True, use_bias=False, layer_name='conv2d_layer_'):
    with tf.variable_scope(layer_name) as scope:
        kernel = tf.get_variable('_kernel', [kernel_size, kernel_size, inputs.get_shape()[-1], out_channels], initializer=tf.truncated_normal_initializer())
        convolution = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding='SAME', name='_conv2d')
        if use_bias:
            bias = tf.get_variable('_bias', [out_channels], initializer=tf.random_normal_initializer())
            convolution = convolution + bias
        if batch_norm:
            batch_mean, batch_variance = tf.nn.moments(convolution, axes=[0, 1, 2])
            convolution = tf.nn.batch_normalization(x=convolution, mean=batch_mean, variance=batch_variance, variance_epsilon=EPSILON, name='_bn', offset=None, scale=None)
    return activation(convolution)

#def conv3d_block(inputs, out_channels, kernel_size=2, stride=1, activation=tf.contrib.keras.layers.LeakyReLU(alpha=0.2), batch_norm=True, use_bias=False, layer_name='conv3d_layer_'):
def conv3d_block(inputs, out_channels, kernel_size=2, stride=1, activation=leaky_relu, batch_norm=True, use_bias=False, layer_name='conv3d_layer_'):
    with tf.variable_scope(layer_name) as scope:
        kernel = tf.get_variable('_kernel', [kernel_size, kernel_size, kernel_size, inputs.get_shape()[-1], out_channels], initializer=tf.truncated_normal_initializer())
        convolution = tf.nn.conv3d(inputs, kernel, [1, stride, stride, stride, 1], padding='SAME', name='_conv3d')
        if use_bias:
            bias = tf.get_variable('_bias', [out_channels], initializer=tf.random_normal_initializer())
            convolution = convolution + bias
        if batch_norm:
            batch_mean, batch_variance = tf.nn.moments(convolution, axes=[0, 1, 2])
            convolution = tf.nn.batch_normalization(x=convolution, mean=batch_mean, variance=batch_variance, variance_epsilon=EPSILON, name='_bn', offset=None, scale=None)
    return activation(convolution)

def conv3d_transpose_block(inputs, out_channels, output_shape, kernel_size=2, stride=1, activation=tf.nn.relu, batch_norm=True, use_bias=False, layer_name='conv3d_transpose_'):
    with tf.variable_scope(layer_name) as scope:
        kernel = tf.get_variable('_kernel', [kernel_size, kernel_size, kernel_size, out_channels, inputs.get_shape()[-1]], initializer=tf.truncated_normal_initializer())
        convolution = tf.nn.conv3d_transpose(inputs, kernel, output_shape, [1, stride, stride, stride, 1], name='_conv3d_transpose')
        if use_bias:
            bias = tf.get_variable('_bias', [out_channels], initializer=tf.random_normal_initializer())
            convolution = convolution + bias
        if batch_norm:
            batch_mean, batch_variance = tf.nn.moments(convolution, axes=[0, 1, 2])
            convolution = tf.nn.batch_normalization(x=convolution, mean=batch_mean, variance=batch_variance, variance_epsilon=EPSILON, name='_bn', offset=None, scale=None)
    return activation(convolution)

def encoder(input_img):
    enc = conv2d_block(inputs=input_img, out_channels=64, kernel_size=11, stride=4, layer_name='enc1')
    enc = conv2d_block(inputs=enc, out_channels=128, kernel_size=5, stride=2, layer_name='enc_2')
    enc = conv2d_block(inputs=enc, out_channels=256, kernel_size=5, stride=2, layer_name='enc_3')
    enc = conv2d_block(inputs=enc, out_channels=512, kernel_size=5, stride=2, layer_name='enc_4')
    enc = conv2d_block(inputs=enc, out_channels=400, kernel_size=8, stride=1, layer_name='enc_5')
    enc =  tf.contrib.layers.flatten(enc)
    z_mean = tf.layers.dense(inputs=enc, units=ENCODING_DIM, kernel_initializer=tf.truncated_normal_initializer(), name='enc_mean')
    z_var = tf.layers.dense(inputs=enc, units=ENCODING_DIM, kernel_initializer=tf.truncated_normal_initializer(), name='enc_var')
    normal_fitted = tf.contrib.distributions.MultivariateNormalDiag(z_mean, z_var)
    sample = normal_fitted.sample(name='enc_sample')
    return normal_fitted, sample, z_mean, z_var
    
def generator(sample, batch_size=BATCH_SIZE):
    gen = tf.layers.dense(inputs=sample, units=2*2*2*512, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(), name='fc')
    gen = tf.reshape(gen, [batch_size, 2, 2, 2, 512])
    gen = conv3d_transpose_block(inputs=gen, out_channels=256, output_shape=[batch_size, 4, 4, 4, 256], kernel_size=4, stride=2, layer_name='gen_2')
    gen = conv3d_transpose_block(inputs=gen, out_channels=128, output_shape=[batch_size, 8, 8, 8, 128], kernel_size=4, stride=2, layer_name='gen_3')
    gen = conv3d_transpose_block(inputs=gen, out_channels=64, output_shape=[batch_size, 16, 16, 16, 64], kernel_size=4, stride=2, layer_name='gen_4')
    gen = conv3d_transpose_block(inputs=gen, out_channels=1, output_shape=[batch_size, 32, 32, 32, 1], kernel_size=4, stride=2, activation=tf.nn.sigmoid, batch_norm=False, layer_name='gen_out')
    gen = tf.reshape(gen, [-1]+OUTPUT_DIMS) 
    return gen 

def discriminator(volume):
    dis = tf.reshape(volume, [-1]+OUTPUT_DIMS+[1])
    dis = conv3d_block(inputs=dis, out_channels=64, kernel_size=4, stride=2, layer_name='dis_1')
    dis = conv3d_block(inputs=dis, out_channels=128, kernel_size=4, stride=2, layer_name='dis_2')
    dis = conv3d_block(inputs=dis, out_channels=256, kernel_size=4, stride=2, layer_name='dis_3')
    dis = conv3d_block(inputs=dis, out_channels=512, kernel_size=4, stride=2, layer_name='dis_4')
    dis = conv3d_block(inputs=dis, out_channels=1, kernel_size=4, stride=1, layer_name='dis_5')
    dis = tf.contrib.layers.flatten(dis)
    dis = tf.layers.dense(inputs=dis, units=1, activation=tf.nn.sigmoid, use_bias=False, kernel_initializer=tf.truncated_normal_initializer(), bias_initializer=tf.random_normal_initializer(), name='dis_out')
    return dis

with tf.variable_scope('encoder') as scope:
    encoded_dist, encoded_sample, encoded_mean, encoded_var = encoder(input_img)
with tf.variable_scope('shared_generator') as scope:
    #generated_from_encoding = generator(encoded_sample)
    generated_from_encoding = generator(encoded_sample)
    scope.reuse_variables()
    generated_from_prior = generator(prior.sample(BATCH_SIZE))
with tf.variable_scope('shared_discriminator') as scope:
    d_x_tilde = discriminator(generated_from_encoding)
    scope.reuse_variables()
    d_x_p = discriminator(generated_from_prior)
    d_x = discriminator(input_volume)
    epsilon = tf.random_uniform(shape=[BATCH_SIZE, 1], minval=0., maxval=1.)
    differences = generated_from_prior - input_volume #input_volume - generated_from_prior
    # interpolates = tf.contrib.layers.flatten(generated_from_prior) + epsilon * tf.contrib.layers.flatten(differences)
    interpolates = tf.contrib.layers.flatten(input_volume) + epsilon * tf.contrib.layers.flatten(differences)
    gradients = tf.gradients(discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), 1, True))
    gradient_penalty = tf.reduce_mean((slopes - 1.)**2)

loss_KL = tf.contrib.distributions.kl(encoded_dist, prior)/40000
loss_recon = tf.sqrt(tf.reduce_sum(tf.square(tf.contrib.layers.flatten(generated_from_encoding) - tf.contrib.layers.flatten(input_volume)), 1))/40000
E_x_tilde = tf.reduce_mean(tf.contrib.layers.flatten(generated_from_encoding), 1)
E_x_p = tf.reduce_mean(tf.contrib.layers.flatten(generated_from_prior), 1)
E_x = tf.reduce_mean(tf.contrib.layers.flatten(input_volume), 1)
loss_critic = tf.reduce_mean(d_x_p) - tf.reduce_mean(d_x) # tf.reduce_mean(d_x - d_x_p) # - E_x_tilde)
loss_wgan = loss_critic + LAMBDA * gradient_penalty
loss_encoder = tf.reduce_mean(loss_KL + loss_recon)
loss_gen = tf.reduce_mean(loss_recon) - tf.reduce_mean(d_x_p)
# loss_total = tf.reduce_mean(loss_wgan + ALPHA_1 * loss_KL + ALPHA_2 * loss_recon) # don't know why they have mentioned this in the paper!!

params_enc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
params_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='shared_generator')
params_dis = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='shared_discriminator')

train_op_enc = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=BETA_1, beta2=BETA_2).minimize(loss_encoder, var_list=params_enc)
train_op_gen = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=BETA_1, beta2=BETA_2).minimize(loss_gen, var_list=params_gen)
train_op_dis = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=BETA_1, beta2=BETA_2).minimize(loss_wgan, var_list=params_dis)

def generate_samples(sess, path, imgs, epoch, threshold=0.9):
    #with tf.variable_scope('encoder', reuse=True):
    #    _, encoded_sample, _, _ = sess.run(encoder(input_img), feed_dict={input_img:imgs})
    with tf.variable_scope('shared_generator', reuse=True):
        _output_volumes = sess.run(generator(encoded_sample, imgs.shape[0]), feed_dict={input_img:imgs})
    _output_volumes[_output_volumes < threshold] = 0
    _output_volumes[_output_volumes >= threshold] = 1
    for i in xrange(imgs.shape[0]):
        np.save(path+'/out_'+str(epoch)+'_'+str(i+1), _output_volumes[i])
        voxel2obj(path+'/out_'+str(epoch)+'_'+str(i+1)+'.obj', _output_volumes[i])
        system('gsutil cp '+path+'/out_'+str(epoch)+'_'+str(i+1)+'.* gs://3dwgan/output1/')

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        data = ShapeNetLoader()
        data.load_data(DATA_DIR)
        print data.get_shapes()
        for i in xrange(EPOCHS):
            loss_e_list = []
            loss_g_list = []
            loss_d_list = []
            it = data.get_batch_iterator(BATCH_SIZE)
            img,vol = it.next()
            img = np.squeeze(img,1)
            print 'EPOCH : ',i
            start = time.time()
            for k in xrange(BATCHES_PER_EPOCH):
                for j in xrange(DISC_ITERS):
                    loss_d,_ = sess.run([loss_wgan, train_op_dis], feed_dict={input_img:img, input_volume:vol})
                    loss_d_list.append(loss_d)
                    img,vol = it.next()
                    img = np.squeeze(img,1)
                loss_k = sess.run(loss_KL, feed_dict={input_img:img, input_volume:vol})
                loss_e,_ = sess.run([loss_encoder, train_op_enc], feed_dict={input_img:img, input_volume:vol})
                loss_e_list.append(loss_e)
                loss_g,_ = sess.run([loss_gen, train_op_gen], feed_dict={input_img:img, input_volume:vol})
                loss_g_list.append(loss_g)
            print 'L_enc : ',np.mean(loss_e_list), '|| L_gen : ', np.mean(loss_g_list), '|| L_dis : ', np.mean(loss_d_list), ' || L_kl : ', loss_k, ' || time: ', time.time()-start
            if (i + 1) % 50 == 0:
                generate_samples(sess, OUT_DIR, img[:1], i+1)

        saver.save(sess, OUT_DIR+'/gan.ckpt')
        system('gsutil cp '+OUT_DIR+'/gan.ckpt* gs://3dwgan/output1/')
