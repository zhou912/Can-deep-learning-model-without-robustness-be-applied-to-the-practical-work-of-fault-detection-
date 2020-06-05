
"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
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
    
adv_fig_defence = np.load("adv_fig_defence.npy").astype(np.float32)
pre_fig_defence = np.load("pre_fig_defence.npy").astype(np.float32)
time_size = 101
cross_size = 401
input_train_size = 25
print (adv_fig_defence)
test = np.zeros(((time_size - input_train_size),(cross_size - input_train_size)))
for i in range((time_size - input_train_size)):
    for j in range((cross_size - input_train_size)):
        if adv_fig_defence[i][j] > 0.5:
            test[i][j] = 1

plt.imshow(test)
plt.show()   
