import os
import numpy as np
from RWoperation import readwriteOperation_pb2 as rwOperation

train_img_name_list = os.listdir('/root/caffework/datafolder/CompanyStandardLabel')
def load_feature():
        fd = rwOperation.read_dict_des( '../datafolder/image_cnn_dict.feature')
        feature_list = []
        for key in fd.keys():
                feature_list.append( fd[key] )
        feature_train = np.array( feature_list, dtype=np.float32 )
        return fd.keys(), feature_train

a = np.load('../datafolder/record.npy')

right_num = 0
for i, x in enumerate(a):
	temp = [y for y in x[0:1000] if y in train_img_name_list]
#	print( i, temp )	
	if temp:
		right_num += 1
	else:
		print( x[0:5] )
print( 'right_num:', right_num )
print('train_na:', len( a ))
print('rate:', float(right_num)/len(a))
