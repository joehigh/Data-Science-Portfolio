from scipy.spatial import distance_matrix
from sklearn.datasets import load_digits
import skimage.transform as skt
import numpy as np
from collections import defaultdict
import pickle
import argparse

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
import random
from sklearn.model_selection import train_test_split, ShuffleSplit

import keras
import tensorflow as tf
import keras.backend as K
from tqdm import tqdm
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from keras.applications.resnet50 import ResNet50

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator
from scipy.spatial.distance import pdist,squareform
from sklearn.cluster import AgglomerativeClustering
import os
import skimage.io as ski
import gc

from random import shuffle
from scipy.stats import entropy

import glob
import pickle as pkl
from PIL import Image

def get_data_path(my_path,ext):
	subdirs = [x[0] for x in os.walk(my_path) if os.path.isdir(x[0])]
	cls_ctr=0
	cls_guide = {}
	images = []
	annotations = []

	for sdir in subdirs[1:]:
		if not sdir.split('/')[-1]=='@eaDir':
			cls_guide[cls_ctr] = sdir.split('/')[-1]
			current_ims = [os.path.join(sdir,f) for f in os.listdir(sdir) if f[-3:]==ext]
			for im in current_ims:
				i = ski.imread(im)
				i = skt.resize(i,[224,224,3])
				images.append(i)
				annotations.append(cls_ctr)
			cls_ctr +=1
	print(cls_guide)
	print(len(images))
	return images,annotations,cls_guide

def prep_x_values(x_train):
	xtrain=np.zeros((len(x_train),224,224,3))
	for i,x in enumerate(x_train):
		xtrain[i,:,:,:] = x
	return xtrain

def prep_y_values(y_train,num_classes=10):
	y_train = keras.utils.np_utils.to_categorical(y_train,num_classes)
	ytrain = np.zeros((len(y_train),num_classes))
	for i,y in enumerate(y_train):
		ytrain[i,:] = y
	return ytrain
	
def build_resnet50_featextract(img_shape=(3,224,224), load_pretrained='/data/resnet50_weights_tf_dim_ordering_tf_kernels.h5'):
	base_model = ResNet50(include_top=True,weights=load_pretrained,input_tensor=None, input_shape=img_shape)
	# Remove classification layer
	base_model.layers.pop()
	# Set output to just end of 2048 layer
	base_model.outputs = [base_model.layers[-1].output]
	base_model.layers[-1].outbound_nodes = []
	return base_model
	
def get_feats(images):
	model = build_resnet50_featextract(img_shape = images[0].shape)
	feats = [model.predict(np.expand_dims(x,axis=0)) for x in images]
	return feats

def build_resnet50_finetune(img_shape=(3,224,224), n_classes=21, load_pretrained='resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', freeze_layers_from='base_model'):
	base_model = ResNet50(include_top=False, weights=load_pretrained, input_tensor=None, input_shape=img_shape)
	x = base_model.output
	x = Flatten()(x)
	predictions = Dense(n_classes, activation='softmax', name='fc{}'.format(n_classes))(x)

	model = Model(input=base_model.input, output=predictions)

	for layer in base_model.layers[:-8]:
		layer.trainable=False
	#for layer in model.layers:
		#print(layer, layer.trainable)

	return model
	
def finetune(X_train,Y_train,X_test,Y_test,n_classes=21,batch_size=64,steps_per_epoch=60000//64, epochs=1):
	model = build_resnet50_finetune(img_shape=X_train[0].shape,n_classes=n_classes)
	#model.summary()
	model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

	train_gen = ImageDataGenerator(rotation_range=8,width_shift_range=0.08, shear_range=0.3, height_shift_range=0.08,zoom_range=0.08)
	test_gen = ImageDataGenerator()

	train_generator = train_gen.flow(X_train,Y_train,batch_size=batch_size)
	test_generator = test_gen.flow(X_test,Y_test,batch_size=batch_size)
	steps_per_epoch = np.ceil(len(X_train)/batch_size)
	print('steps_per_epoch',steps_per_epoch)
	history = model.fit_generator(train_generator,epochs=epochs,validation_data=test_generator,steps_per_epoch=steps_per_epoch) #steps_per_epoch=steps_per_epoch,validation_steps=10000//64)
	return model, history
	
def build_model(shape,num_classes=10):

	model = Sequential()

	model.add(Conv2D(32, (3, 3), padding='same', input_shape=(256,256,3),name='conv1'))
	model.add(Activation('relu',name='relu1'))
	model.add(Conv2D(32, (3, 3),name='conv2'))
	model.add(Activation('relu',name='relu2'))
	model.add(MaxPooling2D(pool_size=(2, 2),name='pool1'))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same',name='conv3'))
	model.add(Activation('relu',name='relu3'))
	model.add(Conv2D(64, (3, 3),name='conv4'))
	model.add(Activation('relu',name='relu4'))
	model.add(MaxPooling2D(pool_size=(2, 2),name='pool2'))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512,name='dense1'))
	model.add(Activation('relu',name='relu5'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes,name='dense2'))
	model.add(Activation('softmax',name='softmax'))

	model.compile(loss='categorical_crossentropy',
				  optimizer=keras.optimizers.SGD(),
				  metrics=['accuracy'])

	#model.summary()
	return model
	
def build_alexnet(shape,num_classes=10,lr=0.01):

	model = Sequential()
	
	model.add(Conv2D(filters=96,input_shape=shape,kernel_size=(11,11),strides=(4,4),padding='valid',name='conv1'))
	model.add(Activation('relu',name='relu1'))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid',name='pool1'))
	
	model.add(Conv2D(filters=256,kernel_size=(11,11),strides=(1,1),padding='valid',name='conv2'))
	model.add(Activation('relu',name='relu2'))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid',name='pool2'))
	
	model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='valid',name='conv3'))
	model.add(Activation('relu',name='relu3'))
	
	model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='valid',name='conv4'))
	model.add(Activation('relu',name='relu4'))
	
	model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='valid',name='conv5'))
	model.add(Activation('relu',name='relu5'))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid',name='pool3'))
	
	model.add(Flatten())
	model.add(Dense(4096,input_shape=(224*224*3,),name='fc1'))
	model.add(Activation('relu',name='relu6'))
	model.add(Dropout(0.4))
	
	model.add(Dense(4096,name='fc2'))
	model.add(Activation('relu',name='relu7'))
	model.add(Dropout(0.4))
	
	model.add(Dense(1000,name='fc1000'))
	model.add(Activation('relu',name='relu1000'))
	model.add(Dropout(0.4))
	
	model.add(Dense(num_classes,name='output'))
	model.add(Activation('softmax',name='softmax'))
	
	model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=lr),metrics=['accuracy'])
	
	model.summary()
	
	return model
	
def train_scratch(model,X_train,y_train,epochs):
	batch_size = 8
	model.reset_states()
	history = model.fit(x=X_train, y=y_train, epochs=epochs, shuffle=True, batch_size=batch_size)
	return model, history