#-*- coding:utf-8 -*-
from __future__ import print_function

import os
#from image_feature import rw_feature
from RWoperation import rwOperation as rw_feature 

#遍历文件夹   
def get_path_dict(imageDir):
    #遍历根目录
    path_dict = dict()
    for root,_,files in os.walk(imageDir):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif']: #bmp、gif、jpg、pic、png、tif               
                filepath = os.path.join(root,file)
                path_dict[file] = filepath
    return path_dict 

def save_image_path(imageDir, save_path): 
    #imageDir : src image path
    #save_path : the proto file save path
    if not os.path.exists(imageDir):
        print('ErrorMessage:', imageDir, ' is not exisit!')
        return -1
    
    path_dict = get_path_dict( imageDir )
    rw_feature.save_dict( path_dict, save_path )
    

def read_image_path( protofile_path ):
    #read data from the protofile
    #protofile_path: the protofile path
    
    if not os.path.exists( protofile_path):
        print('ErrorMessage:', protofile_path, ' is not exisit !!!')
        return -1
    image_path_dict = rw_feature.read_dict( protofile_path )

    return image_path_dict

if __name__ == '__main__':
    imageDir = '../datafolder/CompanyStandardLabel'
    imageDir = '/home/kenneth/ckwork/image_search/LabelImage/allImage1229'
#    print( get_path_dict(rootDir))
    save_path = '../datafolder/test/src_image_path.path'
    save_image_path( imageDir, save_path)

    image_path_dict = read_image_path( save_path )
#    print( image_path_dict.keys() )
#    save_path = './wine_images/src_image_path.txt'
#    save_image_path(rootDir, save_path)

#    image_path_list = read_image_path( save_path )
#    for pa in image_path_list:
#        print(pa)
#    path_list = get_path_list( rootDir)
#    write_data( save_path, path_list)
