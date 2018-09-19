#-*- coding:utf-8 -*-
from __future__ import print_function

import os
import cv2
import numpy as np
import readwriteOperation_pb2 as rwo_pb2

def save_dict_des(des_dict, save_path):
    #save only descriptor, without keypoint
    #des_dict: {image_name:decriptor}
    #save_path: the proto binary file path
    if not des_dict:
        print( 'the input is empty')
        return -1
    feature_des = rwo_pb2.Feature()
    feature_des.name = 'all image feature'

    for key in des_dict.keys():
        one_des = feature_des.feature.add()
        one_des.name = key
        for i in des_dict[key]:
            one_des.des.element.append(i)		
    data = feature_des.SerializeToString()
    with open( save_path, 'w') as f:
        f.write(data)

def read_dict_des(save_path):
    #read only descriptor from proto binary file 
    #save_path: the proto binary file path
    with open( save_path, 'r') as f:
    	data = f.read()
    feature_des = rwo_pb2.Feature()
    feature_des.ParseFromString(data)
#    print( feature_des.name )
    
    feature_dict = {}
    desa = feature_des.feature
    for dd in desa:
        feature_dict[dd.name] = [i for i in dd.des.element]
    return feature_dict
        
def save_feature(image_name, img_kp, img_des, save_path):
     #save surf/sift feature to prototxt file
     #image_name: the image name 
     #img_kp: the KeyPoint list
     #img_des: the descriptors
     #save_path: the save path

    one_feature = rwo_pb2.OneFeature()
    
    one_feature.name = image_name  #保存特征对应的图像名称
    
    #保存keypoint信息
    for kp in img_kp:
        one_kp = one_feature.kps.add()

        one_kp.angle = kp.angle
        one_kp.class_id = kp.class_id
        one_kp.octave = kp.octave
        one_kp.pt.x = kp.pt[0]
        one_kp.pt.y = kp.pt[1]
        one_kp.response = kp.response
        one_kp.size = kp.size
    
    des_shape = np.array( img_des ).shape
    for ds in des_shape:
        one_feature.des.dim.elem.append(ds)

    for x in img_des:
        for y in x:
            one_feature.des.element.append(y)
    
    data = one_feature.SerializeToString()
    
    if os.path.exists(save_path):
        print('errormessage:',save_path,'is already exisits, please change the save path!')
        return -1
    with open(save_path, 'w') as f:
        f.write(data)

def read_feature(data_path):
    #descriptor: read feature from prototxt file 
    #data_path: the prototxt file path

    if not os.path.exists(data_path):
        print('errormessage:',data_path,'is not exisits!')
        return -1
    with open(data_path, 'r') as f:
        data = f.read()

    one_feature = rwo_pb2.OneFeature()
    one_feature.ParseFromString(data)
    
    img_name = one_feature.name
#    print( img_name )

    img_kp = []
    for one_atr in one_feature.kps:
        kp = cv2.KeyPoint()
        kp.angle = one_atr.angle
        kp.class_id = one_atr.class_id
        kp.octave = one_atr.octave
        kp.pt = (one_atr.pt.x, one_atr.pt.y)
        kp.response = one_atr.response
        kp.size = one_atr.size
        img_kp.append(kp)
#    print( img_kp )

    des_shape =  tuple( one_feature.des.dim.elem) 
#    print(des_shape)
    des_data = one_feature.des.element
    img_des = np.array(des_data).reshape(des_shape)
#    print( img_des )
    return img_name, img_kp, img_des

#    return img_kp, img_des

    # kp = image_feature_pb2.KeyPoint()
    # kp.angle = 1.0
    # kp.class_id = 1
    # kp.octave = 3

    # kp.pt.x = 1.0
    # kp.pt.y = 2.0

    # kp.response = 3.0
    # kp.size = 5.0


    # a = [[1,2,3],[3,4,5], [4,5,6]]

    # des = image_feature_pb2.Descriptors()

    # aa = des.array.add()
    # b = a[1][1]
    # aa.element.append(b)
    # aa.element.append(b)

    # print( aa.element)

def save_dict( kv_dict, save_dict_path ):
    # save the path dict, the key is image name, the value is the image/feature path
    #kv_dict: key-value dict, look like {'imag':'osdfjskldfnklvjdaf'}
    #save_dict_path: save path for prototxt file 

    path_dict = rwo_pb2.PathDict()
    for k in kv_dict.keys():
        one_path = path_dict.path_dict.add()
        one_path.key = k
        one_path.value = kv_dict[k]
    
    data = path_dict.SerializeToString()
#    if os.path.exists( save_dict_path ):
#        print('ErrorMessage:', save_dict_path, ' is already exisit!')
#        return -1
    with open(save_dict_path, 'a') as f:
        f.write(data)

def read_dict( dict_path ):
    # read the key-value dict, wich save by save_dict function 
    #dict_path: the prototxt file path
    # get the key-value

    if not os.path.exists( dict_path ):
        print('ErrorMessage:', dict_path, ' is not exisit!')
        return -1
    with open( dict_path, 'r') as f:
        data = f.read()
    
    data_proto = rwo_pb2.PathDict()
    data_proto.ParseFromString(data)

    pdict = {}
    for one_elem in data_proto.path_dict:
        pdict[one_elem.key] = one_elem.value
    return pdict

if __name__ == '__main__':
    kp_aa = cv2.KeyPoint()
    kps_d = [kp_aa,kp_aa]
    des_a = np.array( [[1,2,3], [2,3,4], [5,6,7]])
    save_feature( 'image1.jpg', kps_d, des_a, 'data.txt' ) 
    img_name, img_kp, img_des = read_feature('data.txt')

    print( img_name )
    print( img_kp, type( img_kp ))
    print( img_des, type( img_des ))


#  pdict = dict()
#  pdict['a'] = 'b'
#  pdict['c'] = 'd'
#
#  save_dict_path = 'image.path'
#  save_dict( pdict, save_dict_path )
#  dd = read_dict( save_dict_path )
#  print( dd )





#test3
#    dicte = {}
#    dicte['123'] = [1,2,3,4]
#    dicte['345'] = [12,3,4,56,7,9]
    
#    save_dict_des(dicte, 'feature.cnn_feature')
#    a = read_dict_des('feature.cnn_feature')
#    for key in a.keys():
#        print(key)
#        print( type( a[key]))
#	print( a[key] )



