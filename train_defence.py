from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer
import cyf_datamake as data_pre
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from PIL import Image

from model import Model
from pgd_attack import LinfPGDAttack
with open('config.json') as config_file:
    config = json.load(config_file)


os.chdir("/home/zrs/Desktop/adversarial_attacks")
os.getcwd()

path = '/home/zrs/Desktop/adversarial_attacks'
name = 'poumian.txt'
data_line = data_pre.getLines(path,name)
data_line = data_pre.getAddLines(data_line)
data_label,poumain_list = data_pre.dataMake(data_line,[2000,1400,800])
print (poumain_list)


#训练数据及对应标签
train_X = np.load("train_x.npy").astype(np.float32)
train_Y = np.load("train_y.npy").astype(np.float32)
# 完整的标签数据，用于可视化结果时与预测结果及原始工区进行对比分析
data_label = np.load("data_label.npy")
#原始的工区数据，用于可视化结果时与预测结果及标签进行对比分析
raw_data = np.load("data.npy").astype(np.float32)

input_train_size = 25
half_train_size = int (input_train_size / 2)
'''
数据预处理
'''

#构建训练集与测试集
#train_X = np.reshape(train_X, (-1, input_train_size, input_train_size, 1))
raw_train_y = np.array(train_Y).astype(int)
#将训练集label转换成one hot 形式
train_Y = np.zeros((len(raw_train_y), 2))
train_Y[np.arange(len(raw_train_y)), raw_train_y] = 1

#打散训练数据的顺序
shuffle_indices = np.random.permutation(np.arange(len(train_X)))
train_X = train_X[shuffle_indices]
train_Y = train_Y[shuffle_indices]
batch_size = 16 #每次训练时样本批次大小
model_path = '/home/zrs/Desktop/adversarial_attacks/cnn_model_defence/model'  #模型保存地址
epoch = int(len(train_X) / batch_size) #总的训练次数
print (epoch)
model = Model()

attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])
#模型训练
saver = tf.train.Saver(max_to_keep = 10)
sess = tf.InteractiveSession()

tf.summary.scalar('adv loss',model.total_loss)
merged_summaries = tf.summary.merge_all()
loss_path = '/home/zrs/Desktop/adversarial_attacks/loss_sum'
summary_writer = tf.summary.FileWriter(loss_path,sess.graph)


retrain = False
if retrain:    
    sess.run(tf.global_variables_initializer())
else:
    sess.run(tf.global_variables_initializer())
#    讀取經過對抗訓練的神經網絡
#    saver = tf.train.import_meta_graph('cnn_model_defence/model-1.meta')
#    saver.restore(sess,"./cnn_model_defence/model-1")
##    
#    
#讀取未經過對抗訓練的神經網絡
    model_path_raw = '/home/zrs/Desktop/adversarial_attacks/cnn_model/model.ckpt'  #模型保存地址
    load_path = saver.restore(sess,model_path_raw)

for step in range(5):
#    print (step)
    #epoch = int(epoch / 2)
    
    for i in range (epoch):
        p = (i/epoch)
        print (p,step)
        batch_x = train_X[i * batch_size:(i + 1) * batch_size]
        batch_y = train_Y[i * batch_size:(i + 1) * batch_size]
        batch_x_adv = attack.perturb(batch_x, batch_y, sess)
        
        X_final = np.concatenate([batch_x, batch_x_adv])
        y_final = np.concatenate([batch_y, batch_y])
        
        lr = 0.0001/ (1. + 10 * p)**0.75
#        sess.run(model.optimizer, feed_dict={model.x_input: batch_x, model.y_input: batch_y, model.learning_rate_1 : lr})
        sess.run(model.optimizer, feed_dict={model.x_input: X_final, model.y_input: y_final, model.learning_rate_1 : lr})
        
#        tensorborad_summaries
#        tensorboard_num = i + epoch * step
#        if tensorboard_num % 100 == 0:
#            summary = sess.run(merged_summaries,feed_dict={model.x_input: batch_x_adv, model.y_input: batch_y, model.learning_rate_1 : lr})
#            summary_writer.add_summary(summary,tensorboard_num)
    save_path = saver.save(sess,model_path,global_step = step)



time_size = 101
cross_size = 401
index = poumain_list[9]
test_poumain_y = data_label[9]
#for i in range (1):
#    print (i)
#    index = poumain_list[i]
#    test_poumain = raw_data[index]
#    test_poumain_y = data_label[i]
#    #nomal_train_pre\
#    test_poumian_y_pre =  np.zeros(((time_size),(cross_size)))
#    for j in range(half_train_size,(time_size - half_train_size)):
#        for k in range(half_train_size,(cross_size - half_train_size )):
#            batch_x = np.zeros((input_train_size,input_train_size))
#            batch_x =  raw_data[index,j - half_train_size : j + half_train_size + 1, k - half_train_size : k + half_train_size + 1]
#            batch_x = np.reshape(batch_x,(-1,input_train_size,input_train_size))
#            y_pre = sess.run((model.pre_pro), feed_dict={model.x_input:batch_x})
#            test_poumian_y_pre[j][k] = y_pre
#    
#    plt.subplot(3,1,3)
#    plt.imshow(test_poumian_y_pre)
#    plt.subplot(3,1,2)
#    plt.imshow(test_poumain_y)
#    plt.subplot(3,1,1)
#    plt.imshow(test_poumain)
#    plt.show()   
#    save_fig_path = "/home/zrs/Desktop/adversarial_attacks/poumian/"
#    fig_name =save_fig_path  + str(index)  + ".png"
#    plt.savefig(fig_name)
#
#plt.imshow(raw_data[index])
#plt.show()

#batch_x_set = []
#batch_x_adv_set = []
#batch_y_adv_pre_set = []
#batch_y_pre_set = []
#batch_y_set = []
#for z in range(2):
#    j = (z + 1) * 30
#    for k in range(half_train_size,(cross_size - half_train_size )):
#        test_Y = []
#        batch_x = np.zeros((input_train_size,input_train_size))
#        batch_x = raw_data[index,j - half_train_size : j + half_train_size + 1, k - half_train_size : k + half_train_size + 1]
#        batch_x = np.reshape(batch_x,(-1,input_train_size,input_train_size))
#        y = test_poumain_y[j][k]
#        test_Y.append(y)
#        raw_test_y = np.array(test_Y).astype(int)
#        batch_y = np.zeros((len(raw_test_y), 2))
#        batch_y[np.arange(len(raw_test_y)), raw_test_y] = 1
#        batch_x_adv = attack.perturb(batch_x, batch_y, sess)
#        y_pre = sess.run((model.pre_pro), feed_dict={model.x_input:batch_x})
#        y_pre_adv = sess.run((model.pre_pro), feed_dict={model.x_input:batch_x_adv})
#        batch_x_set.append(batch_x)
#        batch_x_adv_set.append(batch_x_adv)
#        batch_y_adv_pre_set.append(y_pre_adv)
#        batch_y_pre_set.append(y_pre)
#        batch_y_set.append(y)
#batch_x_set = np.array(batch_x_set)
#batch_x_adv_set = np.array(batch_x_adv_set)
#batch_y_adv_pre_set = np.array(batch_y_adv_pre_set)
#batch_y_pre_set = np.array(batch_y_pre_set)
#batch_y_set = np.array(batch_y_set)
#np.save("batch_x_set.npy",batch_x_set) 
#np.save("batch_x_adv_set.npy",batch_x_adv_set) 
#np.save("batch_y_adv_pre_set.npy",batch_y_adv_pre_set) 
#np.save("batch_y_pre_set.npy",batch_y_pre_set) 
#np.save("batch_y_set.npy",batch_y_set) 
#print ('over')


#all_poumian


index = poumain_list[9]
test_poumain = raw_data[index]
test_poumain_y = data_label[9]
#nomal_train_pre\
test_poumian_y_pre =  np.zeros(((time_size),(cross_size)))
for j in range(half_train_size,(time_size - half_train_size)):
    for k in range(half_train_size,(cross_size - half_train_size )):
        batch_x = np.zeros((input_train_size,input_train_size))
        batch_x =  raw_data[index,j - half_train_size : j + half_train_size + 1, k - half_train_size : k + half_train_size + 1]
        batch_x = np.reshape(batch_x,(-1,input_train_size,input_train_size))
        y_pre = sess.run((model.pre_pro), feed_dict={model.x_input:batch_x})
        test_poumian_y_pre[j][k] = y_pre

test_poumian_y_adv_pre =  np.zeros(((time_size),(cross_size)))
deal_poumain_x = raw_data[index]
for j in range(half_train_size,(time_size - half_train_size)):
    for k in range(half_train_size,(cross_size - half_train_size )):
        test_Y = []
        batch_x = np.zeros((input_train_size,input_train_size))
        batch_x =  raw_data[index,j - half_train_size : j + half_train_size + 1, k - half_train_size : k + half_train_size + 1]
        batch_x = np.reshape(batch_x,(-1,input_train_size,input_train_size))
        y = test_poumain_y[j][k]
        test_Y.append(y)
        raw_test_y = np.array(test_Y).astype(int)
        batch_y = np.zeros((len(raw_test_y), 2))
        batch_y[np.arange(len(raw_test_y)), raw_test_y] = 1
        batch_x_adv = attack.perturb(batch_x, batch_y, sess)
        y_pre_adv = sess.run((model.pre_pro), feed_dict={model.x_input:batch_x_adv})
        test_poumian_y_adv_pre[j][k] = y_pre_adv



raw_fig = np.zeros(((time_size - input_train_size),(cross_size - input_train_size)))
pre_fig = np.zeros(((time_size - input_train_size),(cross_size - input_train_size)))
adv_fig = np.zeros(((time_size - input_train_size),(cross_size - input_train_size)))
raw_fig = raw_data[index,half_train_size:(time_size-half_train_size),half_train_size:(cross_size-half_train_size)]
pre_fig = test_poumian_y_pre[half_train_size:(time_size-half_train_size),half_train_size:(cross_size-half_train_size)]
adv_fig = test_poumian_y_adv_pre[half_train_size:(time_size-half_train_size),half_train_size:(cross_size-half_train_size)]
np.save("adv_fig_defence.npy",adv_fig) 
np.save("pre_fig_defence.npy",pre_fig) 
plt.subplot(3,1,3)
plt.imshow(adv_fig)
plt.subplot(3,1,2)
plt.imshow(pre_fig)
plt.subplot(3,1,1)
plt.imshow(raw_fig)
plt.show()   

