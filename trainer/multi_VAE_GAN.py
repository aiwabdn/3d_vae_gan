from __future__ import with_statement
from __future__ import absolute_import

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation,Dense,Reshape,Flatten
from keras.layers.convolutional import Cropping3D
from keras_contrib.layers.convolutional import Deconvolution3D
from keras.layers.convolutional import Convolution3D,Convolution2D,Conv2D,MaxPooling2D
from keras.layers import concatenate,Lambda,ZeroPadding2D,AveragePooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D
from keras import layers
from keras.engine.topology import get_source_inputs
import numpy as np
import os
from os import system
import random
from keras.utils import generic_utils

from lib.read import ShapeNetLoader
from lib.voxel import voxel2obj
import time

dim1=200
dim2=200
dim3=1

encoding_dim=200

# TOTAL EPOCHS
nb_epoch = 701
n_batch_per_epoch = 44
BATCH_SIZE=10

# no. of dis iteration for each gen iteration
disc_iter = 5

# GRADIENT PENALTY MULTIPLIER AS USED IN  IMPROVED WGAN
LAMBDA = 10


epoch_size = n_batch_per_epoch * BATCH_SIZE

# TRAINING DATASET DIRECTOREY
directory = directory='/home/sam/04256520/'

output_dim=32*32*32



sess = tf.Session()

# PASS TENSORFLOW SESSION TO KERAS
K.set_session(sess) 

# SET LEARNING PHASE TO TRAIN
K.set_learning_phase(1)


# Keras Lambda layer to calculate row wise max of a 2D tensor
def column_max(x):
    return K.max(x,1)

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None):




   
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
            
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.add a reshape as layer tensorflow
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')


    return model



def save_model_weights(encoder_model,generator_model, discriminator_model, e):

    model_path = "../../models/3DWGAN_airplanes1/" 

    if e % 25 == 0:
        enc_weights_path = os.path.join(model_path, '%s_epoch%s.h5' % (encoder_model.name, e))
        encoder_model.save_weights(enc_weights_path, overwrite=True)
        
        gen_weights_path = os.path.join(model_path, '%s_epoch%s.h5' % (generator_model.name, e))
        generator_model.save_weights(gen_weights_path, overwrite=True)

        disc_weights_path = os.path.join(model_path, '%s_epoch%s.h5' % (discriminator_model.name, e))
        discriminator_model.save_weights(disc_weights_path, overwrite=True)
        


def load_image( infilename ) :
    img = color.rgb2gray(io.imread(infilename))
    img = transform.resize(img,(200,200)) 
    data = np.expand_dims( img,axis=-1 )
    
    return data
    
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def batch_generator(BATCH_SIZE,directory,train=True,views=3):
    
    if not train:
        directory = directory[:-1]+"_test/"
    dir_list = get_immediate_subdirectories(directory)
    ids = random.sample(range(0, len(dir_list)), BATCH_SIZE)
    img_data =[]
    _3d_data = []
    
    for folder_id in ids:
        
        folder_path = directory+dir_list[folder_id]+'/screenshots/'
        onlyfiles = [f for f in listdir(folder_path) if (isfile(join(folder_path, f)))]
        
        onlyfiles= list(filter(lambda a: (a.endswith("4.png") or a.endswith("9.png") or a.endswith("7.png")), onlyfiles))
        
        #print(onlyfiles)
        img_views=[]
        for view in xrange(views):
            image_path = folder_path + onlyfiles[view]
            img_views.append(load_image(image_path))
            
        
        #print(image_path)
        _3d = np.load(directory+dir_list[folder_id]+'/model.npy')
        
        img_data.append(img_views)
        _3d_data.append(_3d)
        
    img_data =  np.asarray(img_data)
   
    return img_data[:,0],img_data[:,1],img_data[:,2],np.asarray(_3d_data)
        

def inf_train_gen(BATCH_SIZE,directory):
    while True:
        img_view1,img_view2,img_view3,_3d = batch_generator(BATCH_SIZE,directory)
        yield img_view1,img_view2,img_view3,_3d

            
            



def sample_noise(batch_size):

    return np.random.normal(0,1, size=[batch_size,200,1,1,1])

def multiview_encoder(encoder_input1,encoder_input2,encoder_input3,encoder_input4, encoder_input5):

    base = ResNet50(input_shape=(dim1,dim2,dim3))
    #base.load_weights("/home/sam/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
    
    process1 = base(encoder_input1)
    process2 = base(encoder_input2)
    process3 = base(encoder_input3)
    process4 = base(encoder_input4)
    process5 = base(encoder_input5)
 
    
    conc = concatenate([process1,process2,process3,process4,process5], axis=-1)
    
    reshape = Reshape((5,2048))(conc)#tf.reshape(conc,[-1,4,2048]) # reshape into a 200*1*1*1 array
    
    
    view_pool = Lambda(column_max)(reshape)
    
    fc = Dense(2048,activation='relu')(view_pool)
    
    z_mean = Dense(encoding_dim, activation='relu')(fc)
    z_log_var = Dense(encoding_dim, activation='relu')(fc)
    
    
    return z_mean,z_log_var


        
def encoder(encoder_input):
    
    #h = Reshape((1,137,137))(encoder_input)
    h = Convolution2D(filters=64,kernel_size=11,padding='same',input_shape=(dim1,dim2,dim3),strides= 4,kernel_initializer ="he_normal")(encoder_input)
    h = BatchNormalization(axis=-1)(h)
    h = Activation('relu')(h)
    
    h = Convolution2D(filters=128,kernel_size=5,padding='same',strides = 2,kernel_initializer ="he_normal")(h)
    h = BatchNormalization(axis=-1)(h)
    h = Activation('relu')(h)
    
    h = Convolution2D(filters=256,kernel_size=5,padding='same',strides= 2,kernel_initializer ="he_normal")(h)
    h = BatchNormalization(axis=-1)(h)
    h = Activation('relu')(h)
    
    
    h = Convolution2D(filters=512,kernel_size=5,padding='same',strides= 2,kernel_initializer ="he_normal")(h)
    h = BatchNormalization(axis=-1)(h)
    h = Activation('relu')(h)
    
    
    h = Convolution2D(filters=400,kernel_size=8,padding='same',strides= 1,kernel_initializer ="he_normal")(h)
    h = BatchNormalization(axis=-1)(h)
    h = Activation('relu')(h)
    
    h = Flatten()(h)
    
    z_mean = Dense(200,kernel_initializer ="he_normal")(h)
    z_log_var = Dense(200,kernel_initializer ="he_normal")(h)

    return z_mean,z_log_var

def discriminator(discriminator_input):
    
    h = Reshape((32,32,32,1))(discriminator_input)
    
    h = Convolution3D(filters=64, kernel_size=4,padding='same',use_bias=False, strides = 2,kernel_initializer ="he_normal")(h)
    #h = BatchNormalization(mode=2, axis=1)(h)
    h = LeakyReLU(0.2)(h)
    
    h = Convolution3D(filters=128, kernel_size=4,padding='same',use_bias=False, strides = 2,kernel_initializer ="he_normal")(h)#(h)
    #h = BatchNormalization(mode=2, axis=1)(h)
    h = LeakyReLU(0.2)(h)
   
    h = Convolution3D(filters=256, kernel_size=4,padding='same',use_bias=False, strides = 2,kernel_initializer ="he_normal")(h)
    #h = BatchNormalization(mode=2, axis=1)(h)
    h = LeakyReLU(0.2)(h)
   
    
    h = Convolution3D(filters=512, kernel_size=4, padding='same',use_bias=False, strides = 2,kernel_initializer ="he_normal")(h)
    #h = BatchNormalization(mode=2, axis=1)(h)
    h = LeakyReLU(0.2)(h)
    
    
    h = Convolution3D(filters=1, kernel_size=3, padding="same", use_bias=False,kernel_initializer ="he_normal")(h)

    #h = Dense(1024,activation='relu') (h)
    discriminator_output = Dense(1,kernel_initializer ="he_normal")(h)
    
    return discriminator_output



def generator(generator_input):
    
    
    h = Deconvolution3D(filters=384, kernel_size=(4, 4, 4),padding="valid", output_shape=(None, 4, 4, 4, 384),use_bias=False)(generator_input)
    
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
     #h = Dropout(0.2)(h)
    
    h = Deconvolution3D(filters=192, kernel_size=(4, 4, 4),strides=(2, 2, 2),output_shape=(None, 10, 10, 10, 192) ,padding='valid',use_bias=False)(h)
    h = Cropping3D(cropping=((1, 1), (1, 1), (1, 1)))(h)
    h = BatchNormalization(axis=-1)(h)
    h = Activation('relu')(h)
     #h = Dropout(0.2)(h)
    
    
    h = Deconvolution3D(filters=96, kernel_size=(4, 4, 4),strides=(2, 2, 2), output_shape=(None, 18, 18, 18, 96),padding='valid',use_bias=False)(h)
    h = Cropping3D(cropping=((1, 1), (1, 1), (1, 1)))(h)
    h = BatchNormalization(axis=-1)(h)
    h = Activation('relu')(h)
     #h = Dropout(0.2)(h)
    
    h = Deconvolution3D(filters=48, kernel_size=(4, 4, 4),strides=(2, 2, 2), output_shape=(None, 34, 34, 34, 48),padding='valid',use_bias=False)(h)
    h = Cropping3D(cropping=((1, 1), (1, 1), (1, 1)))(h)
    h = BatchNormalization(axis=-1)(h)
    h = Activation('relu')(h)
     #h = Dropout(0.2)(h) 
    
    h = Deconvolution3D(filters=1, kernel_size=(3, 3, 3),padding='same',output_shape=(None, 32, 32, 32, 1),use_bias=False)(h)
    
    generator_output = Activation('tanh')(h)
    
    h= Flatten()(generator_output)
    
    return h


    
    
def inference(image_data1,image_data2,image_data3,image_data4,image_data5,real_3d_data):
    """ Connections b/w different models"""
    
    
    z_p = tf.random_normal((BATCH_SIZE, 1,1,1,200), 0, 1) # normal dist for GAN
    eps = tf.random_normal((BATCH_SIZE, 200), 0, 1) # normal dist for VAE

 
    
    ### ENCODER                      
    encoder_input1 = Input(shape=[dim1,dim2,dim3])
    encoder_input2 = Input(shape=[dim1,dim2,dim3])
    encoder_input3 = Input(shape=[dim1,dim2,dim3])
    encoder_input4 = Input(shape=[dim1,dim2,dim3])
    encoder_input5 = Input(shape=[dim1,dim2,dim3])
   
    
    enc_mean,enc_sigma  = multiview_encoder(encoder_input1,encoder_input2,encoder_input3,encoder_input4,encoder_input5)
    e_net = Model(inputs=[encoder_input1,encoder_input2,encoder_input3,encoder_input4,encoder_input5], outputs=[enc_mean,enc_sigma],name="encoder")

    z_x_mean, z_x_log_sigma_sq = e_net(inputs=[image_data1,image_data2,image_data3,image_data4, image_data5]) 
   
    

 
    
    ### GENERATOR                                 
    generator_input  = Input(shape = [1,1,1,200])
    generator_output = generator(generator_input)
    g_net = Model(inputs=generator_input, outputs=generator_output,name="generator")
    z_x = z_x_mean + K.exp(z_x_log_sigma_sq / 2) * eps # get actual values of z from mean and variance
    z_x = tf.reshape(z_x,[BATCH_SIZE,1,1,1,200]) # reshape into a 200*1*1*1 array
    
    
    x_p = g_net(z_p)   # output from noise
    x_tilde = g_net(z_x)  # output 3d model of the image
    
    
    ### DISCRIMINATOR
    discriminator_input = Input(shape = [32,32,32,1])
    d = discriminator(discriminator_input)   
    d_net = Model(inputs=discriminator_input, outputs=d,name="discriminator")
    
   
    
    
    #_, l_x_tilde = discriminator(x_tilde)
    d_x =   d_net(real_3d_data) 
    d_x_p = d_net(x_p)
    
      
        
    return e_net,g_net,d_net,x_tilde,z_x_mean, z_x_log_sigma_sq, z_x, x_p, d_x, d_x_p, z_p


def loss(z_x_log_sigma_sq, z_x_mean,x_tilde, d_x, d_x_p, dim1, dim2, dim3,real_3d_data,x_p):
    """
    Loss functions for  KL divergence, Discrim, Generator, Lth Layer Similarity
    """
    
    
    KL_loss = (- 0.5 * K.sum(1 + z_x_log_sigma_sq - K.square(z_x_mean) - K.exp(z_x_log_sigma_sq), axis=-1))/dim1/dim2/dim3
    
    # Discriminator Loss
    D_loss = tf.reduce_mean(d_x_p) - tf.reduce_mean(d_x)
    # Generator Loss    
    G_loss = -tf.reduce_mean(d_x_p)
    
                                
    # Reconstruction loss
    Reconstruction_loss = tf.reduce_sum(tf.square(x_tilde-tf.reshape(real_3d_data,[BATCH_SIZE,-1])))/dim1/dim2/dim3
                                       
                                     
                           
    alpha = tf.random_uniform(
    shape=[BATCH_SIZE,1],
    minval=0.,maxval=1.)


    
    differences = x_p-tf.reshape(real_3d_data,[BATCH_SIZE,-1])
    
    interpolates = tf.reshape(real_3d_data,[BATCH_SIZE,-1]) + (alpha*differences)
    
    
    gradients = tf.gradients(d_net(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
                                
    return  KL_loss, D_loss, G_loss, Reconstruction_loss,gradient_penalty
    
    
    
# define input placeholders
image_data1 = tf.placeholder(tf.float32, shape=[BATCH_SIZE,dim1,dim2,dim3])
image_data2 = tf.placeholder(tf.float32, shape=[BATCH_SIZE,dim1,dim2,dim3])
image_data3 = tf.placeholder(tf.float32, shape=[BATCH_SIZE,dim1,dim2,dim3])
image_data4 = tf.placeholder(tf.float32, shape=[BATCH_SIZE,dim1,dim2,dim3])
image_data5 = tf.placeholder(tf.float32, shape=[BATCH_SIZE,dim1,dim2,dim3])



real_3d_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE,32,32,32])

# define network structure
e_net,g_net,d_net,x_tilde,z_x_mean, z_x_log_sigma_sq, z_x, x_p, d_x, d_x_p, z_p = inference(image_data1,image_data2,image_data3,image_data4,image_data5,real_3d_data)


# define individual losses
KL_loss, D_loss, G_loss, Reconstruction_loss , gradient_penalty= loss(z_x_log_sigma_sq, z_x_mean,x_tilde, d_x, d_x_p, dim1, dim2, dim3,real_3d_data,x_p)


## add up losses for each part
L_e = KL_loss + Reconstruction_loss

L_g = Reconstruction_loss + G_loss
L_d = D_loss + LAMBDA* gradient_penalty 


# get trainable weights
E_params = e_net.trainable_weights
G_params = g_net.trainable_weights
D_params = d_net.trainable_weights




# define optimizers
enc_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-4, 
    beta1=0.5,
    beta2=0.9
).minimize(L_e, var_list=E_params)

gen_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-4, 
    beta1=0.5,
    beta2=0.9
).minimize(L_g, var_list=G_params)

disc_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-4, 
    beta1=0.5, 
    beta2=0.9
).minimize(L_d, var_list=D_params)


# define iterator



# generates samples each x epochs
eps = tf.random_normal((BATCH_SIZE, 200), 0, 1)
z_x_sample_mean,z_x_sample_log_sigma_sq = e_net(inputs=[image_data1,image_data2,image_data3,image_data4,image_data5])

z_x_sample = z_x_sample_mean + K.exp(z_x_sample_log_sigma_sq / 2) * eps
z_x_sample = tf.reshape(z_x_sample,[BATCH_SIZE,1,1,1,200])

fixed_image_samples = g_net(z_x_sample)

def generate_sample(epoch, image_data1,image_data2,image_data3,image_data4,image_data5, _3d_batch, threshold=0.9):

    samples = session.run(fixed_image_samples,feed_dict= {image_data1:img_view1,image_data2:img_view2,image_data3:img_view3, image_data4:img_view4,image_data5:img_view5})

    Reconstruction_loss1 = np.sum(np.square(samples-np.reshape(_3d_batch,[BATCH_SIZE,-1])))/dim1/dim2/dim3

    np.save('/tmp/out_train_'+str(epoch), samples[BATCH_SIZE-1])
    output = np.reshape(samples[BATCH_SIZE-1], (32,32,32))
    output[output < threshold] = 0
    output[output >= threshold] = 1
    voxel2obj('/tmp/out_train_'+str(epoch)+'.obj', np.reshape(samples[BATCH_SIZE-1], (32,32,32)))
    system('gsutil cp /tmp/out_train_'+str(epoch)+'.* gs://3dwgan/output_multivaegan/')

    #img_batch1,_3d_batch1 = batch_generator(BATCH_SIZE ,directory,train=False)

    #samples = session.run(fixed_image_samples,feed_dict= {image_data:img_batch1})
    #Reconstruction_loss2 = np.sum(np.square(samples-np.reshape(_3d_batch1,[BATCH_SIZE,-1])))/dim1/dim2/dim3


    print "\ntrain :"+str(Reconstruction_loss1)#+" test:"+str(Reconstruction_loss2)+"for epoch "+str(epoch)

    #np.save('/tmp/out_test_'+str(epoch), samples[BATCH_SIZE-1])
    #voxel2obj('/tmp/out_test_'+str(epoch)+'.obj', np.reshape(samples[BATCH_SIZE-1], (32,32,32)))
    #system('gsutil cp /tmp/out_test_'+str(epoch)+'.npy gs://3dwgan/output_vaegan/')

def generate_sample1(epoch):
    img_view1,img_view2,img_view3,_3d_batch = batch_generator(BATCH_SIZE ,directory)


    samples = session.run(fixed_image_samples,feed_dict= {image_data1:img_view1,image_data2:img_view2,image_data3:img_view3})
    
    Reconstruction_loss1 = np.sum(np.square(samples-np.reshape(_3d_batch,[BATCH_SIZE,-1])))/dim1/dim2/dim3
    
    
    img_view1,img_view2,img_view3,_3d_batch = batch_generator(BATCH_SIZE ,directory,train=False)

    samples = session.run(fixed_image_samples,feed_dict= {image_data1:img_view1,image_data2:img_view2,image_data3:img_view3})
    Reconstruction_loss2 = np.sum(np.square(samples-np.reshape(_3d_batch,[BATCH_SIZE,-1])))/dim1/dim2/dim3
    
                                 
    print("\ntrain :"+str(Reconstruction_loss1)+" test:"+str(Reconstruction_loss2)+"for epoch "+str(epoch))
    

    if not os.path.exists('../../results_airplane1/'):
        os.makedirs('../../results_airplane1/')
    np.save('../../results_airplane1/'+str(epoch)+"_"+'.npy',samples)
    np.save('../../results_airplane1/'+str(epoch)+'gt1_.npy',img_view1)
    np.save('../../results_airplane1/'+str(epoch)+'gt2_.npy',img_view2)
    np.save('../../results_airplane1/'+str(epoch)+'gt3_.npy',img_view3)
    np.save('../../results_airplane1/'+str(epoch)+'3d_.npy',_3d_batch)
    

    #np.save('../../results_airplane1/gt_image_.npy',img_batch)

if __name__=='__main__':
    with sess as session:
        saver = tf.train.Saver()

        session.run(tf.global_variables_initializer())
        

        x = ShapeNetLoader()
        x.load_data(directory)
        print x.get_shapes()

        #################
        # Start training
        ################
        for e in range(nb_epoch):
            # kernel_initializerialize progbar and batch counter
            it = x.get_batch_iterator(BATCH_SIZE)
            start = time.time()
            list_enc_loss = []
            list_gen_loss = []
            list_dis_loss = []
            
            for batch_counter in xrange(n_batch_per_epoch):
                img_data, _3d_data = it.next()
                img_view1 = img_data[:,0,:,:,:]
                img_view2 = img_data[:,1,:,:,:]
                img_view3 = img_data[:,2,:,:,:]
                img_view4 = img_data[:,3,:,:,:]
                img_view5 = img_data[:,4,:,:,:]
                
                for d in xrange(disc_iter):
                    
                    
                    dis_cost, _ = session.run([L_d, disc_train_op],feed_dict={real_3d_data :_3d_data})
                    list_dis_loss.append(dis_cost)
                    img_data, _3d_data = it.next()
                    img_view1 = img_data[:,0,:,:,:]
                    img_view2 = img_data[:,1,:,:,:]
                    img_view3 = img_data[:,2,:,:,:]
                    img_view4 = img_data[:,3,:,:,:]
                    img_view5 = img_data[:,4,:,:,:]


                enc_cost, _ = session.run([L_e, enc_train_op],feed_dict= {image_data1:img_view1,image_data2:img_view2,image_data3:img_view3, image_data4:img_view4,image_data5:img_view5, real_3d_data:_3d_data})
                list_enc_loss.append(enc_cost)
                
                
                gen_cost, _ = session.run([L_g, gen_train_op],feed_dict={real_3d_data :_3d_data,image_data1:img_view1,image_data2:img_view2,image_data3:img_view3, image_data4:img_view4,image_data5:img_view5})
                list_gen_loss.append(gen_cost)
                            

            print "Loss_E ", np.mean(list_enc_loss)," Loss_G ", np.mean(list_gen_loss)," Loss_D ", np.mean(list_dis_loss)
            if (e+1)%50 == 0:
                img_data, _3d_data = x.get_test_set(BATCH_SIZE)
                img_view1 = img_data[:,0,:,:,:]
                img_view2 = img_data[:,1,:,:,:]
                img_view3 = img_data[:,2,:,:,:]
                img_view4 = img_data[:,3,:,:,:]
                img_view5 = img_data[:,4,:,:,:]
                generate_sample(e+1, image_data1,image_data2,image_data3,image_data4,image_data5, _3d_data)

            if (e+1)%200 == 0:
                saver.save(session, '/tmp/vaegan'+str(e+1)+'.ckpt')
                system('gsutil cp /tmp/vaegan'+str(e+1)+'.ckpt* gs://3dwgan/output_multivaegan/')

            print('\nEpoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))
            
            
        saver.save(session, '/tmp/vaeganfinal.ckpt')
        system('gsutil cp /tmp/vaeganfinal.ckpt* gs://3dwgan/output_multivaegan')
