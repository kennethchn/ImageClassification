#!/usr/bin/env python
#-*- coding:utf-8 -*-

import time
import numpy as np
import cv2

import cnn_search

name_list , feature_train = cnn_search.load_feature()

image_path = '../datafolder/image/demo.jpg'

net = cnn_search.load_net()

sf = cv2.BFMatcher()

st_time = time.time()
one_feature = cnn_search.one_extract_feature( net, image_path )
print( one_feature )
print( type( one_feature ))
one_feature = one_feature[np.newaxis, :]

res = sf.knnMatch( one_feature, feature_train, k = 1000 )
print( 'spend time :', time.time() - st_time )
#print(res )
dest_name_list = []
for idx in res:
	dest_name_list.append( name_list[idx.trainIdx] )
	

import feature_extract

surf = feature_extract.create_detector('surf')
img = cv2.imread(image_path )
img = cv2.resize(img, (360,360))
kp, des = feature_extract.detect( surf, image)

from RWoperation import rwOperation 
image_feature_path = '../datafolder/test/src_image_feature_path.path'
feature_path_dict = rwOperation.read_dict( image_feature_path )

f_train_path = []
for nameIdx in dest_name_list:
	f_train_path.append(feature_path_dict[nameIdx])

f_train_des = []
svm_clf = rwOperation.ana_svm()
for pathIdx in f_train_path:
	img_name, img_kp, img_des = read_feature( pathIdx )
	 
	r_good_match, good_match =rwOperation.match(kp, des.astype(np.float32),img_kp, img_des.astype(np.float32), 0.75, 'bf')
	if svm_clf.predict([[float( len(ransanc_gmatch)), float( len(gmatch))]]):
		print( img_name )
