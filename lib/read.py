from __future__ import with_statement
from __future__ import absolute_import
import numpy as np
import lib.binvox_rw as bvx 
from os import system
import tarfile
import re
from io import BytesIO
from PIL import Image
from itertools import ifilter

class ShapeNetLoader(object):

    def __init__(self, n_imgs=1, desired_imgs=['04'], input_img_sz=(200,200,1), input_voxel_sz=(32,32,32)):
        self.n_images = n_imgs
        self.desired_imgs = desired_imgs
        self.train_out = []
        self.train_set = []
        self.input_img_sz = input_img_sz
        self.input_voxel_sz = input_voxel_sz

    def load_image(self, filebuffer):
        img = Image.open(BytesIO(bytearray(filebuffer.read())))
        img = img.convert('L').resize(self.input_img_sz[:2])
        return np.reshape(np.asarray(img, dtype='float32'), self.input_img_sz)

    def load_data(self, bucket='gs://shapenet_data', target_obj_set=['04256520']):
        print 'downloading data'
        #system('gsutil cp gs://shapenet_data/images4791415.npy /tmp/')
        #system('gsutil cp gs://shapenet_data/voxels.npy /tmp/')
        system('gsutil cp gs://shapenet_data/sofas5.tar.gz /tmp/')
        print 'downloading done'
        system('tar xvzf /tmp/sofas5.tar.gz -C /tmp/')
        self.data_path='/tmp/'
        self.object_set = target_obj_set
        self.train_set = np.load('/tmp/train_in.npy')
        self.test_set = np.load('/tmp/test_in.npy')
        self.train_out = np.load('/tmp/train_voxels.npy')
        self.test_out = np.load('/tmp/test_voxels.npy')

    def load_data1(self, bucket='gs://shapenet', target_obj_set=['04256520']):
        self.data_path = bucket
        self.object_set = target_obj_set
        with tarfile.open(self.data_path+'/ShapenetVox32.tar', 'r') as voxtar, tarfile.open(self.data_path+'/ShapeNetRendering.tar','r') as imgtar:
            for obj in self.object_set:
                print obj
                regex_voxel = re.compile('.*/'+obj+'/.*/model\.binvox')
                names = list(ifilter(regex_voxel.match, voxtar.getnames()))
                pic_locs = ['ShapeNetRendering/'+x.partition('/')[-1].rpartition('/')[0]+'/rendering/' for x in names]
                for i,j in enumerate(names):
                    self.train_out.append(np.reshape(np.array(bvx.read_as_3d_array(voxtar.extractfile(j)).data, dtype='float32'), self.input_voxel_sz))
                    loaded_imgs = []
                    for m in self.desired_imgs:
                        loaded_imgs.append(self.load_image(imgtar.extractfile(pic_locs[i]+m+'.png')))
                    self.train_set.append(loaded_imgs)
        self.train_set = np.asarray(self.train_set)
        self.train_out = np.asarray(self.train_out)

    def get_shapes(self):
        return self.train_set.shape, self.test_set.shape, self.train_out.shape

    def get_batch_iterator(self, batch_size=5):
        l = len(self.train_set)
        for ndx in xrange(0, l, batch_size):
            yield self.train_set[ndx:min(ndx + batch_size, l)], self.train_out[ndx:min(ndx + batch_size, l)]

    def get_test_set(self, batch_size=5):
        return self.test_set[:batch_size], self.test_out[:batch_size]

    def save(self):
        np.save(self.data_path+'/images', self.train_set)
        np.save(self.data_path+'/voxels', self.train_out)
