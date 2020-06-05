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

time_size = 101
cross_size = 401
index = poumain_list[9]
test_poumain_y = data_label[9]







