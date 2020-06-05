"""
針對於幾個樣本點（非整個剖面）
分析對抗樣本與原始樣本之間在圖像上的差異
"""
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
import scipy.io as io

from model import Model
from pgd_attack import LinfPGDAttack
with open('config.json') as config_file:
    config = json.load(config_file)

batch_x_set = np.load("batch_x_set.npy").astype(np.float32)
batch_x_adv_set = np.load("batch_x_adv_set.npy").astype(np.float32)
batch_y_adv_pre_set = np.load("batch_y_adv_pre_set.npy").astype(np.float32)
batch_y_pre_set = np.load("batch_y_pre_set.npy").astype(np.float32)
batch_y_set = np.load("batch_y_set.npy").astype(np.float32)

#accuracy 
num_batch_x_set = batch_y_set.shape[0]
#print (num_batch_x_set,'num_batch_x_set')
#原始訓練的準確率
ture_pre_num_normal = 0
for i in range(num_batch_x_set):
    if batch_y_pre_set[i] > 0.5 and batch_y_set[i] == 1:
        ture_pre_num_normal = ture_pre_num_normal + 1
    elif batch_y_pre_set[i] < 0.5 and batch_y_set[i] == 0:
        ture_pre_num_normal = ture_pre_num_normal + 1 
normal_acc = ture_pre_num_normal / num_batch_x_set         
print ('normal_accuracy:',normal_acc)

ture_pre_num_adv = 0
for i in range(num_batch_x_set):
    if batch_y_adv_pre_set[i] > 0.5 and batch_y_set[i] == 1:
        ture_pre_num_adv = ture_pre_num_adv + 1
    elif batch_y_adv_pre_set[i] < 0.5 and batch_y_set[i] == 0:
        ture_pre_num_adv = ture_pre_num_adv + 1 
adv_acc = ture_pre_num_adv / num_batch_x_set         
print ('adv_accuracy:',adv_acc)

#fault_point_index 斷層樣本點下標的提取
fault_point_index = []
for i in range (num_batch_x_set):
    if batch_y_set[i] == 1:
        fault_point_index.append(i)
#print ('fault_point_index:',fault_point_index)

batch_size = 25

batch_x_output = []
batch_y_output = []
#fault_data_save 斷層點的存儲
for i in range(len(fault_point_index)):
    batch_x_point = np.zeros(shape = (3,batch_size,batch_size))
    index = fault_point_index[i]
    batch_x_point[0] = batch_x_set[index][0]   #原始數據 
    batch_x_point[1] = batch_x_adv_set[index][0] #對抗數據
    noise = batch_x_adv_set[index][0] - batch_x_set[index][0] 
    batch_x_point[2] = noise     #原始數據與對抗數據之間的差異
    batch_x_output.append(batch_x_point)
    
    batch_y_point = np.zeros(shape = (3))
    batch_y_point[0] = 1
    batch_y_point[1] = batch_y_pre_set[index]  #正常預測值
    batch_y_point[2] = batch_y_adv_pre_set[index] #對抗樣本預測值
    batch_y_output.append(batch_y_point)    
    
    print (index,'index')
    print (batch_y_set[index],'label')
    print (batch_y_pre_set[index],'normal_pre')
    print (batch_y_adv_pre_set[index],'adv_pre')
    noise = batch_x_set[index][0] - batch_x_adv_set[index][0]
    plt.subplot(3,1,1)
    plt.imshow(batch_x_set[index][0],cmap = 'gray')
    plt.subplot(3,1,2)
    plt.imshow(batch_x_adv_set[index][0],cmap = 'gray')
    plt.subplot(3,1,3)
    plt.imshow(noise,cmap = 'gray')
    plt.show() 
    save_fig_path = "/home/zrs/Desktop/adversarial_attacks/fault_point/"
    fig_name =save_fig_path +  "point" + str(index)  + ".png"
    plt.savefig(fig_name)
batch_x_output = np.array(batch_x_output)
batch_y_output = np.array(batch_y_output)
io.savemat('batch_x_output',{'batch_x_output':batch_x_output})
io.savemat('batch_y_output',{'batch_y_output':batch_y_output})


#所有預測點結果的存儲
batch_x_output_all = []
batch_y_output_all = []
for i in range(num_batch_x_set):
    #print (i)
    index = i
    batch_x_point = np.zeros(shape = (3,batch_size,batch_size))
    batch_x_point[0] = batch_x_set[index][0] 
    batch_x_point[1] = batch_x_adv_set[index][0]
    noise = batch_x_adv_set[index][0] - batch_x_set[index][0]
    batch_x_point[2] = noise
    batch_x_output_all.append(batch_x_point)
    
    batch_y_point = np.zeros(shape = (3))
    batch_y_point[0] = batch_y_set[index]
    batch_y_point[1] = batch_y_pre_set[index]
    batch_y_point[2] = batch_y_adv_pre_set[index]
    batch_y_output_all.append(batch_y_point)    
    
    
#    print (batch_y_set[index],'label')
#    print (batch_y_pre_set[index],'normal_pre')
#    print (batch_y_adv_pre_set[index],'adv_pre')
#    noise = batch_x_set[index][0] - batch_x_adv_set[index][0]
#    plt.subplot(3,1,1)
#    plt.imshow(batch_x_set[index][0],cmap = 'gray')
#    plt.subplot(3,1,2)
#    plt.imshow(batch_x_adv_set[index][0],cmap = 'gray')
#    plt.subplot(3,1,3)
#    plt.imshow(noise,cmap = 'gray')
#   #plt.show() 
#    save_fig_path = "/home/zrs/Desktop/adversarial_attacks/all_point/"
#    fig_name =save_fig_path +  "point" + str(index)  + ".png"
#    plt.savefig(fig_name)  
#    
batch_x_output_all = np.array(batch_x_output_all)
batch_y_output_all = np.array(batch_y_output_all)
io.savemat('all_batch_x_output',{'all_batch_x_output':batch_x_output_all})
io.savemat('all_batch_y_output',{'all_batch_y_output':batch_y_output_all})
#print (batch_x_output.shape)
#print(batch_y_output.shape)