#coding=utf-8

#使用保存好的模型

import tensorflow as tf
import numpy as np
import  gender_train_data as train_data
import  matplotlib.pyplot as plt
import  os


np.set_printoptions(suppress=True)

#取一张图片
input_image = train_data.images[0:1]
labels = train_data.labels[0:1]
fig2,ax2 = plt.subplots(figsize=(2,2))
ax2.imshow(np.reshape(input_image, (112, 92,3)))
plt.show()


sess = tf.Session()
graph_path=os.path.abspath('./model/my-gender-v1.0.meta')
model=os.path.abspath('./model/')

server = tf.train.import_meta_graph(graph_path)
server.restore(sess,tf.train.latest_checkpoint(model))

graph = tf.get_default_graph()

#填充feed_dict
x = graph.get_tensor_by_name('input_images:0')
y = graph.get_tensor_by_name('input_labels:0')
feed_dict={x:input_image,y:labels}


#第一层卷积+池化
relu_1 = graph.get_tensor_by_name('relu_1:0')
max_pool_1 = graph.get_tensor_by_name('max_pool_1:0')

#第二层卷积+池化
relu_2 = graph.get_tensor_by_name('relu_2:0')
max_pool_2 = graph.get_tensor_by_name('max_pool_2:0')

#第三层卷积+池化
relu_3 = graph.get_tensor_by_name('relu_3:0')
max_pool_3 = graph.get_tensor_by_name('max_pool_3:0')

#全连接最后一层输出
f_softmax = graph.get_tensor_by_name('f_softmax:0')


#relu_1_r,max_pool_1_,relu_2,max_pool_2,relu_3,max_pool_3,f_softmax=sess.run([relu_1,max_pool_1,relu_2,max_pool_2,relu_3,max_pool_3,f_softmax],feed_dict)



#----------------------------------各个层特征可视化-------------------------------




#conv1 特征
r1_relu = sess.run(relu_1,feed_dict)
r1_tranpose = sess.run(tf.transpose(r1_relu,[3,0,1,2]))
fig,ax = plt.subplots(nrows=1,ncols=16,figsize=(16,1))
for i in range(16):
    ax[i].imshow(r1_tranpose[i][0])
plt.title('Conv1 16*112*92')
plt.show()

#pool1特征
max_pool_1 = sess.run(max_pool_1,feed_dict)
r1_tranpose = sess.run(tf.transpose(max_pool_1,[3,0,1,2]))
fig,ax = plt.subplots(nrows=1,ncols=16,figsize=(16,1))
for i in range(16):
    ax[i].imshow(r1_tranpose[i][0])
plt.title('Pool1 16*56*46')
plt.show()


#conv2 特征
r2_relu = sess.run(relu_2,feed_dict)
r2_tranpose = sess.run(tf.transpose(r2_relu,[3,0,1,2]))
fig,ax = plt.subplots(nrows=1,ncols=32,figsize=(32,1))
for i in range(32):
    ax[i].imshow(r2_tranpose[i][0])
plt.title('Conv2 32*56*46')
plt.show()

#pool2 特征
max_pool_2 = sess.run(max_pool_2,feed_dict)
tranpose = sess.run(tf.transpose(max_pool_2,[3,0,1,2]))
fig,ax = plt.subplots(nrows=1,ncols=32,figsize=(32,1))
for i in range(32):
    ax[i].imshow(tranpose[i][0])
plt.title('Pool2 32*28*23')
plt.show()


#conv3 特征
r3_relu = sess.run(relu_3,feed_dict)
tranpose = sess.run(tf.transpose(r3_relu,[3,0,1,2]))
fig,ax = plt.subplots(nrows=1,ncols=64,figsize=(32,1))
for i in range(64):
    ax[i].imshow(tranpose[i][0])
plt.title('Conv3 64*28*23')
plt.show()

#pool3 特征
max_pool_3 = sess.run(max_pool_3,feed_dict)
tranpose = sess.run(tf.transpose(max_pool_3,[3,0,1,2]))
fig,ax = plt.subplots(nrows=1,ncols=64,figsize=(32,1))
for i in range(64):
    ax[i].imshow(tranpose[i][0])
plt.title('Pool3 64*14*12')
plt.show()

print sess.run(f_softmax,feed_dict)