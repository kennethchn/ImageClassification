#-*-coding:utf-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf 

def produceData(r,w,d,num):
    r1 = r-w/2
    r2 = r+w/2
    #上半圆
    theta1 =  np.random.uniform(0, np.pi ,num)
    X_Col1 = np.random.uniform( r1*np.cos(theta1),r2*np.cos(theta1),num)[:, np.newaxis]
    X_Row1 = np.random.uniform(r1*np.sin(theta1),r2*np.sin(theta1),num)[:, np.newaxis]
    Y_label1 = np.ones(num) #类别标签为1
    #下半圆
    theta2 = np.random.uniform(-np.pi, 0 ,num)
    X_Col2 = (np.random.uniform( r1*np.cos(theta2),r2*np.cos(theta2),num) + r)[:, np.newaxis]
    X_Row2 = (np.random.uniform(r1 * np.sin(theta2), r2 * np.sin(theta2), num) -d)[:,np.newaxis]
    Y_label2 = -np.ones(num) #类别标签为-1,注意：由于采取双曲正切函数作为激活函数，类别标签不能为0
    #合并
    X_Col = np.vstack((X_Col1, X_Col2))
    X_Row = np.vstack((X_Row1, X_Row2))
    X = np.hstack((X_Col, X_Row))
    Y_label = np.hstack((Y_label1,Y_label2))
    Y_label.shape = (num*2 , 1)
    return X,Y_label

if __name__ == '__main__':
    x, label = produceData(10, 180, 20, 100)
    # print(x)
    # print(label)
    data_2d = np.array(x)

    meta_path = './parameter_model/model.ckpt.meta'
    model_path = './parameter_model/model.ckpt'

    saver = tf.train.import_meta_graph(meta_path)

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        saver.restore(sess, model_path)
        graph = tf.get_default_graph()
        prob_op = graph.get_operation_by_name('prediction')
        p = graph.get_tensor_by_name('prediction:0')
        print( sess.run(p, feed_dict={xs:data_2d}))