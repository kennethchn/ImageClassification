import os
import cv2
import time 
import feature_extract 
from image_feature import rw_feature 


feature_root_path = '/home/kenneth/ckwork/image_search/image_feature'

images_name_list = os.listdir('original_wine_images')

surf_detector = feature_extract.create_detector()
# for image_name in images_name_list:
#     image_path = os.path.join('original_wine_images', image_name)
#     image = cv2.imread(image_path)
#     image = cv2.resize( image, (256,256))
#     start_time = time.time()
#     kp, des = fe.detect( surf_detector, image )
    
#     print( time.time() - start_time, len(kp) )

image_o = cv2.imread(os.path.join('original_wine_images', images_name_list[0]))
image = cv2.resize( image_o, (256, 256 ))
start_time = time.time()
kp, des = feature_extract.detect( surf_detector, image )

filename, extension = os.path.splitext( images_name_list[0] )
data_save_filename = filename + '.surf'
#import os
#file_path = "D:/test/test.py"
#(filepath,tempfilename) = os.path.split(file_path)
#(filename,extension) = os.path.splitext(tempfilename)
#
rw_feature.save_feature(images_name_list[0], kp, des, data_save_filename)

print( time.time() - start_time )