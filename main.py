#-*- coding:utf-8 -*-
from __future__ import print_function

import os
import numpy as np 

import image_path

import feature_extract

import feature_match

imageDir = '/home/kenneth/ckwork/image_search/image_add'
save_root_path = './image_add_floder'
if not os.path.exists(save_root_path):
    os.mkdir(save_root_path)
src_image_path_save_path = os.path.join(save_root_path, 'src_image_path.path') #保存原图像路径词典
image_path.save_image_path( imageDir, src_image_path_save_path)

feature_extract.save_feature( src_image_path_save_path, save_root_path )

feature_match.class_image(save_root_path)