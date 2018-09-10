#-*-coding:utf-8 -*-
from __future__ import print_function

import os
import cv2
import image_path

img_path = 'companyTestLabel'
path_dict = image_path.get_path_dict(img_path)

save_path = 'image_path_add'
if not os.path.exists(save_path):
    os.mkdir(save_path)

index = 0
for key in path_dict.keys():
    index += 1
    img = cv2.imread(path_dict[key])
    cv2.imwrite(os.path.join(save_path, 'img_' + str(index).zfill(6)+'.jpg'), img)