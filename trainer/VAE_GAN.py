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
from keras.layers.convolutional import Convolution3D,Convolution2D
import numpy as np
import os
from os import system
import random
from keras.utils import generic_utils

import tensorflow as tf
import numpy as np
from lib.read import ShapeNetLoader
from lib.voxel import voxel2obj
import time

# INPUT IMAGE DIMENSIONS

dim1=200
dim2=200
dim3=1



# TOTAL EPOCHS
nb_epoch = 701
n_batch_per_epoch = 9
BATCH_SIZE=50

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
#K.set_image_data_format('channels_first')

# SET LEARNING PHASE TO TRAIN
K.set_learning_phase(1)


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
    data = np.asarray( img, dtype="float32" )
    return data

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def batch_generator(BATCH_SIZE, directory, train=True):
    img,vol = it.next()
    print 'batch_generator: generated batch ',img.shape, ', ', vol.shape
    return img,vol

def batch_generator1(BATCH_SIZE,directory,train=True):

    if not train:
        directory = directory[:-1]+"_test/"
        dir_list = get_immediate_subdirectories(directory)
        ids = random.sample(range(0, len(dir_list)), BATCH_SIZE)
        img_data =[]
        _3d_data = []

    for folder_id in ids:
        image_id = random.sample(range(0, 4),1)
        folder_path = directory+dir_list[folder_id]+'/screenshots/'
        onlyfiles = [f for f in listdir(folder_path) if (isfile(join(folder_path, f)))]

        onlyfiles= list(filter(lambda a: (a.endswith("4.png") or a.endswith("6.png") or a.endswith("7.png") or a.endswith("8.png")), onlyfiles))
        #print(onlyfiles)
        image_path = folder_path + onlyfiles[image_id[0]]
        #print(image_path)
        _3d = np.load(directory+dir_list[folder_id]+'/model.npy')
        img = load_image(image_path)
        img_data.append([img])
        _3d_data.append([_3d])

    return np.asarray(img_data),np.squeeze(np.asarray(_3d_data),axis=1)


def inf_train_gen(BATCH_SIZE,directory):
    while True:
        print 'inf_train_gen : generating batch'
        img,_3d = batch_generator(BATCH_SIZE,directory)
        yield img,_3d






def sample_noise(batch_size):

    return np.random.normal(0,1, size=[batch_size,200,1,1,1])



def encoder(encoder_input):

    #h = Reshape((1,137,137))(encoder_input)
    h = Convolution2D(filters=64,kernel_size=11,padding='same',input_shape=(200, 200,1),strides= 4,kernel_initializer ="he_normal")(encoder_input)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)

    h = Convolution2D(filters=128,kernel_size=5,padding='same',strides = 2,kernel_initializer ="he_normal")(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)

    h = Convolution2D(filters=256,kernel_size=5,padding='same',strides= 2,kernel_initializer ="he_normal")(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)


    h = Convolution2D(filters=512,kernel_size=5,padding='same',strides= 2,kernel_initializer ="he_normal")(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)


    h = Convolution2D(filters=400,kernel_size=8,padding='same',strides= 1,kernel_initializer ="he_normal")(h)
    h = BatchNormalization(axis=1)(h)
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




def inference(image_data,real_3d_data):
    """ Connections b/w different models"""


    z_p = tf.random_normal((BATCH_SIZE, 1,1,1,200), 0, 1) # normal dist for GAN
    eps = tf.random_normal((BATCH_SIZE, 200), 0, 1) # normal dist for VAE



    ### ENCODER                      
    encoder_input = Input(shape = [dim1,dim2,dim3])
    enc_mean,enc_sigma  = encoder(encoder_input)
    e_net = Model(inputs=encoder_input, outputs=[enc_mean,enc_sigma],name="encoder")
    z_x_mean, z_x_log_sigma_sq = e_net(image_data) # get z from the input   


    ### GENERATOR                                 
    generator_input  = Input(shape = [1,1,1,200])
    generator_output = generator(generator_input)
    g_net = Model(inputs=generator_input, outputs=generator_output,name="generator")
    z_x = z_x_mean + K.exp(z_x_log_sigma_sq / 2) * eps # get actual values of z from mean and variance

    z_x = tf.reshape(z_x,[BATCH_SIZE,1,1,1,200]) # reshape into a 1*1*1*200 array
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
image_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE,dim1,dim2,dim3])
real_3d_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE,32,32,32])



# define network structure
e_net,g_net,d_net,x_tilde,z_x_mean, z_x_log_sigma_sq, z_x, x_p, d_x, d_x_p, z_p = inference(image_data,real_3d_data)

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
#img_batch,_3d_batch = batch_generator(BATCH_SIZE ,directory)

# generates samples each x epochs
eps = tf.random_normal((BATCH_SIZE, 200), 0, 1)
z_x_sample_mean,z_x_sample_log_sigma_sq = e_net(image_data)

z_x_sample = z_x_sample_mean + K.exp(z_x_sample_log_sigma_sq / 2) * eps
z_x_sample = tf.reshape(z_x_sample,[BATCH_SIZE,1,1,1,200])

fixed_image_samples = g_net(z_x_sample)

def generate_sample(epoch, img_batch, _3d_batch, threshold=0.9):

    samples = session.run(fixed_image_samples,feed_dict= {image_data:img_batch})

    Reconstruction_loss1 = np.sum(np.square(samples-np.reshape(_3d_batch,[BATCH_SIZE,-1])))/dim1/dim2/dim3

    np.save('/tmp/out_train_'+str(epoch), samples[BATCH_SIZE-1])
    output = np.reshape(samples[BATCH_SIZE-1], (32,32,32))
    output[output < threshold] = 0
    output[output >= threshold] = 1
    voxel2obj('/tmp/out_train_'+str(epoch)+'.obj', np.reshape(samples[BATCH_SIZE-1], (32,32,32)))
    system('gsutil cp /tmp/out_train_'+str(epoch)+'.* gs://3dwgan/output_vaegan/')

    #img_batch1,_3d_batch1 = batch_generator(BATCH_SIZE ,directory,train=False)

    #samples = session.run(fixed_image_samples,feed_dict= {image_data:img_batch1})
    #Reconstruction_loss2 = np.sum(np.square(samples-np.reshape(_3d_batch1,[BATCH_SIZE,-1])))/dim1/dim2/dim3


    print "\ntrain :"+str(Reconstruction_loss1)#+" test:"+str(Reconstruction_loss2)+"for epoch "+str(epoch)

    #np.save('/tmp/out_test_'+str(epoch), samples[BATCH_SIZE-1])
    #voxel2obj('/tmp/out_test_'+str(epoch)+'.obj', np.reshape(samples[BATCH_SIZE-1], (32,32,32)))
    #system('gsutil cp /tmp/out_test_'+str(epoch)+'.npy gs://3dwgan/output_vaegan/')


def generate_sample1(epoch):

    samples = session.run(fixed_image_samples,feed_dict= {image_data:img_batch})

    Reconstruction_loss1 = np.sum(np.square(samples-np.reshape(_3d_batch,[BATCH_SIZE,-1])))/dim1/dim2/dim3


    img_batch1,_3d_batch1 = batch_generator(BATCH_SIZE ,directory,train=False)

    samples = session.run(fixed_image_samples,feed_dict= {image_data:img_batch1})
    Reconstruction_loss2 = np.sum(np.square(samples-np.reshape(_3d_batch1,[BATCH_SIZE,-1])))/dim1/dim2/dim3


    print("\ntrain :"+str(Reconstruction_loss1)+" test:"+str(Reconstruction_loss2)+"for epoch "+str(epoch))


    if not os.path.exists('../../results_airplane1/'):
        os.makedirs('../../results_airplane1/')
        np.save('../../results_airplane1/'+str(epoch)+"_"+'.npy',samples)
        np.save('../../results_airplane1/gt_.npy',_3d_batch)
        np.save('../../results_airplane1/gt_image_.npy',img_batch)



if __name__ == '__main__':
    with sess as session:
        saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())

        # loading data
        x = ShapeNetLoader()
        x.load_data(directory)
        print x.get_shapes()


        #################
        # Start training
        ################
        for e in range(nb_epoch):
            # kernel_initializerialize progbar and batch counter

            it = x.get_batch_iterator(BATCH_SIZE)
            #progbar = generic_utils.Progbar(epoch_size)
            start = time.time()
            list_enc_loss = []
            list_gen_loss = []
            list_dis_loss = []

            #while batch_counter < n_batch_per_epoch:
            for batch_counter in xrange(n_batch_per_epoch):
                _2d_data,_3d_data = it.next()
                _2d_data = _2d_data[:,4,:,:,:]
                for d in xrange(disc_iter):
                    dis_cost, _ = session.run([L_d, disc_train_op],feed_dict={real_3d_data :_3d_data})
                    list_dis_loss.append(dis_cost)
                    _2d_data,_3d_data = it.next()
                    _2d_data = _2d_data[:,4,:,:,:]

                enc_cost, _ = session.run([L_e, enc_train_op],feed_dict= {image_data: _2d_data , real_3d_data:_3d_data})
                list_enc_loss.append(enc_cost)


                gen_cost, _ = session.run([L_g, gen_train_op],feed_dict={real_3d_data :_3d_data,image_data: _2d_data})
                list_gen_loss.append(gen_cost)

            print "Loss_E ", np.mean(list_enc_loss)," Loss_G ", np.mean(list_gen_loss)," Loss_D ", np.mean(list_dis_loss)
            if (e+1)%50 == 0:
                _2d_data,_3d_data = it.next()
                _2d_data = _2d_data[:,4,:,:,:]
                generate_sample(e, _2d_data, _3d_data)
                saver.save(session, '/tmp/vaegan'+str(e)+'.ckpt')
                system('gsutil cp /tmp/vaegan'+str(e)+'.ckpt* gs://3dwgan/output_vaegan/')

            print '\nEpoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start)


            # Save model weights (by default, every 5 epochs)
            #save_model_weights(e_net,g_net, d_net, e)
        saver.save(session, '/tmp/vaeganfinal.ckpt')
        system('gsutil cp /tmp/vaeganfinal.ckpt* gs://3dwgan/output_vaegan')
