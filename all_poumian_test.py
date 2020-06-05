#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:37:46 2019
預測的整個剖面，生成mat文件方便在MATLAB上面進行對比
@author: zrs
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

os.chdir("/home/zrs/Desktop/adversarial_attacks")
os.getcwd()

path = '/home/zrs/Desktop/adversarial_attacks'
name = 'poumian.txt'
data_line = data_pre.getLines(path,name)
data_line = data_pre.getAddLines(data_line)
data_label,poumain_list = data_pre.dataMake(data_line,[2000,1400,800])
train_X = np.load("train_x.npy").astype(np.float32)
train_Y = np.load("train_y.npy").astype(np.float32)
# 完整的标签数据，用于可视化结果时与预测结果及原始工区进行对比分析
data_label = np.load("data_label.npy")
#原始的工区数据，用于可视化结果时与预测结果及标签进行对比分析
raw_data = np.load("data.npy").astype(np.float32)
#預測結果讀取
adv_fig = np.load("adv_fig.npy")
pre_fig = np.load("pre_fig.npy")
index = poumain_list[9]

input_train_size = 25
half_train_size = int (input_train_size / 2)
time_size = 101
cross_size = 401

#將原始數據與標籤切割成預測結果相同的大小，因爲預測結果沒有預測邊界部分
raw_fig = raw_data[index,half_train_size:(time_size - half_train_size),half_train_size:(cross_size - half_train_size)]
label_fig = data_label[9,half_train_size:(time_size - half_train_size),half_train_size:(cross_size - half_train_size)]
io.savemat('pre_fig',{'pre_fig':pre_fig})
io.savemat('adv_fig',{'adv_fig':adv_fig})
io.savemat('label_fig',{'label_fig':label_fig})
io.savemat('raw_fig',{'raw_fig':raw_fig})
#plt.imshow(adv_fig)
#plt.show()


'''
all_poumian_acc
'''
x_range = adv_fig.shape[0] #預測結果在x方向上的點數
y_range = adv_fig.shape[1] #預測結果在y方向上的點數
total_num = x_range * y_range
pre_ture_normal = 0
pre_ture_adv = 0
#正常預測的準確率
for i in range(x_range):
    for j in range(y_range):
        if label_fig[i][j] == 0 and pre_fig[i][j] < 0.5:
            pre_ture_normal = pre_ture_normal + 1
        elif label_fig[i][j] == 1 and pre_fig[i][j] > 0.5:
            pre_ture_normal = pre_ture_normal + 1
normal_acc = pre_ture_normal / total_num
print ('normal_acc',normal_acc)
#對抗訓練後預測正確的準確率
for i in range(x_range):
    for j in range(y_range):
        if label_fig[i][j] == 0 and adv_fig[i][j] < 0.5:
            pre_ture_adv = pre_ture_adv + 1
        elif label_fig[i][j] == 1 and adv_fig[i][j] > 0.5:
            pre_ture_adv = pre_ture_adv + 1
adv_acc = pre_ture_adv / total_num
print ('adv_acc',adv_acc)
#plt.subplot(3,1,3)
#plt.imshow(adv_fig)
#plt.subplot(3,1,2)
#plt.imshow(pre_fig)
#plt.subplot(3,1,1)
#plt.imshow(label_fig)
#plt.show()   

'''
all_poumian_fooling rate
'''
fool_num = 0
for i in range(x_range):
    for j in range(y_range):
        if pre_fig[i][j] < 0.5 and adv_fig[i][j] > 0.5:
            fool_num = fool_num + 1
        elif pre_fig[i][j] > 0.5 and adv_fig[i][j] < 0.5:
            fool_num = fool_num + 1
fool_rate = fool_num / total_num
print('fooling rate',fool_rate)

