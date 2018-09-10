#-*- coding:utf-8 -*-
from __future__ import print_function

import numpy as np 

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import tensorflow as tf

def get_ana_data():
    #分析数据得到最优阈值
    with open('data_ana_foranaly.txt', 'r') as f:
        data = f.readlines()

    ana_data = []
    for one_data in data:
        a = one_data.strip().split('\t')
        ana_data.append( [float(a[0]), float(a[1]), int(a[5])] )
    return ana_data

def ana_svm():

    data = np.array( get_ana_data())
    data_2d = data[:,0:2]
    label = data[:,2].astype(np.float64)
    
    svm_clf = Pipeline(( ("scaler", StandardScaler()),
                        ("linear_svc", LinearSVC(C=1, loss="hinge")) ,))

    svm_clf.fit( data_2d, label )
    return svm_clf

def svm_demo():
    import numpy as np
    from sklearn import datasets
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    iris = datasets.load_iris()
    X = iris["data"][:, (2,3)]
    y = (iris["target"] == 2).astype( np.float64 )

    svm_clf = Pipeline(( ("scaler", StandardScaler()),
                        ("linear_svc", LinearSVC(C=1, loss="hinge")) ,))

    svm_clf.fit( X, y )
    res = svm_clf.predict( [[5.5, 1.7]] )

    print( res )

def add_layer(layername, inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    with tf.variable_scope(layername, reuse=None):
        Weights = tf.get_variable("weights", shape=[in_size, out_size],
                                  initializer= tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", shape=[1,out_size],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


if __name__ == '__main__':

    svm_clf = ana_svm()
    res = svm_clf.predict( [[5.5, 1.7]] )
    print( res )

#    svm_demo()

    # data = np.array( get_ana_data())
    # data_2d = data[:,0:2]
    # label = data[:,2:3]

    # # h = len(data_2d)

    # # x_data_2d = tf.placeholder( shape=[h,2], dtype = tf.float32 )

    # # w = tf.Variable( tf.random_normal(shape = [2,1]))
    # # b = tf.Variable( tf.random_normal(shape=[1,1])) 

    # # result = tf.matmul( x_data_2d, w ) + b

    # ###define placeholder for inputs to network
    # xs = tf.placeholder(tf.float32, [None, 2])
    # ys = tf.placeholder(tf.float32, [None, 1])
    # ###添加隐藏层
    # l1 = add_layer("layer1",xs, 2, 20, activation_function=tf.tanh)
    # ###添加输出层
    # prediction = add_layer("layer2",l1, 20, 1, activation_function=tf.tanh)
    # ###MSE 均方误差
    # loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
    # ###优化器选取 学习率设置 此处学习率置为0.1
    # train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    # ###tensorflow变量初始化，打开会话
    # init = tf.global_variables_initializer()#tensorflow更新后初始化所有变量不再用tf.initialize_all_variables()
    # sess = tf.Session()
    # sess.run(init)

    # for i in range(1000):
    #     sess.run(train_step, feed_dict={xs:data_2d, ys:label})
    #     the_loss = sess.run(loss, feed_dict={xs:data_2d, ys:label})

    #     print(the_loss)
    
    # saver_path = './parameter_model/model.ckpt'
    # saver = tf.train.Saver()
    # config = tf.ConfigProto()

    # saved_path = saver.save( sess, saver_path)

 
     

