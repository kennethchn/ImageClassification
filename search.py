#!/usr/bin/env python
#-*- coding:utf-8 -*-
from __future__ import print_function
import os
import sys
import copy
sys.path.append('/root/caffe/python')
sys.path.append('/root/caffework/RWoperation')
import time
import cv2
import caffe
import numpy as np

import rwOperation 

caffe.set_mode_gpu()
caffe.set_device(0)

def load_net():
	weight = '/root/caffework/resnet/ResNet-152-model.caffemodel'
	deploy_file = '/root/caffework/resnet/ResNet_152_deploy.prototxt'

	net = caffe.Net( deploy_file, weight, caffe.TEST )
	return net
def one_extract_feature(image_path):
	net = load_net()
	transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_raw_scale('data', 255)
	transformer.set_channel_swap('data', (2,1,0))

	net.blobs['data'].reshape(1,3,224,224)
	
	try:
		image = caffe.io.load_image( image_path )
	except:
		print('read image error, error path:', image_path)

	transformered_image = transformer.preprocess( 'data', image )
	net.blobs['data'].data[...] = transformered_image 
	st0 = time.time()
	net.forward()
	print( time.time() - st0 )
	one_feature = np.squeeze( net.blobs['pool5'].data)
	return one_feature 

def extract_feature( image_path):
	net = load_net()
	transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_raw_scale('data', 255)
	transformer.set_channel_swap('data', (2,1,0))

	net.blobs['data'].reshape(1,3,224,224)
	
	features_dict = {} 
	image_list = os.listdir(image_path)
	for img in image_list:
		image_full_path = os.path.join( image_path, img )
		try:
			image = caffe.io.load_image( image_full_path )
		except:
			print('read image error, error path:', image_full_path)
			continue
		transformered_image = transformer.preprocess( 'data', image )
		net.blobs['data'].data[...] = transformered_image 
		st0 = time.time()
		net.forward()
		print( time.time() - st0 )
		features_dict[img] = copy.deepcopy( np.squeeze( net.blobs['pool5'].data))
	return features_dict 

def load_feature():
	fd = rwOperation.read_dict_des( '../image_cnn_dict.feature')
	feature_list = []
	for key in fd.keys():
		feature_list.append( fd[key] )
	feature_train = np.array( feature_list, dtype=np.float32 )
	return fd.keys(), feature_train

def search_demo(feature, feature_train):
	sf = cv2.BFMatcher()
	res = sf.knnMatch( feature, feature_train, k = 1000 )
	print( res )
	return res 
if __name__ == '__main__':
#test1
#	image_path = './test_resnet_feature'
#	image_path = './image'
#	net = load_net()
# 	feature_dict = extract_feature(net, image_path )	
#	rwOperation.save_dict_des( feature_dict, 'image_cnn_dict.feature')

#test2
#	fd = rwOperation.read_dict_des( 'image_cnn_dict.feature')
#	print( fd.keys() )
#	for key in fd.keys():
#		print( fd[key][0] )
#		print( len(fd[key]) )
#		print( type( fd[key]))	

#test3
#	image_path = './image/demo.jpg'
#	afeature = one_extract_feature( image_path)
#	afeature = afeature[np.newaxis, :]		
#	print( afeature.shape, afeature.dtype )
#	
#	name_list, feature_train = load_feature()
#	print(feature_train.shape, feature_train.dtype )
#	match_result = search_demo( afeature, feature_train )
#	for mr in match_result:
#		print( name_list[mr[0].trainIdx])
	
#test4
	image_path = '/root/caffework/TestLabelImage'
#	image_path = './image'
	search_feature = extract_feature( image_path )
	name_list, feature_train = load_feature()
	
	print( name_list[0:3])
	print( feature_train[0:3])	

	record_result = []
	for key in search_feature.keys():
		afeature = search_feature[key]
		afeature = afeature[np.newaxis, :]
		print( afeature )
		match_result = search_demo( afeature, feature_train )

		temp_record = []
		temp_record.append( key )
		for mr in  match_result[0] :
			temp_record.append( name_list[mr.trainIdx] )
		record_result.append( copy.deepcopy(temp_record) )	
	print( record_result )
	np.save( '../record.npy', record_result )


