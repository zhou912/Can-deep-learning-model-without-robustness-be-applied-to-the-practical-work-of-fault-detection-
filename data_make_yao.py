import cyf_datamake as data_pre
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
'''
姚老師數據的樣本生成
input_train_size 爲輸入樣本的大小
'''
path = 'F:/123/adversarial_attacks'
name = 'poumian.txt'
data_line = data_pre.getLines(path,name)
data_line = data_pre.getAddLines(data_line)
data_label,poumain_list = data_pre.dataMake(data_line,[2000,1400,800])
print (poumain_list)
#下一步将这些数据构建成数据集
#数据归一化，便于处理
def data_normalize(data):
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data

inline_size = 401
time_size = 101
cross_size = 401
raw_data = data_normalize(io.loadmat('seismicdata.mat')['seismicdata'])
raw_data = raw_data.transpose(0,2,1)
new_raw_data = np.zeros(shape=(inline_size,time_size,cross_size))
for i in range(inline_size):
    for j in range(time_size):
        for k in range(cross_size):
            new_raw_data[i][j][k] = raw_data[i][j][cross_size - k - 1]
#data_normalize
new_raw_data = data_normalize(new_raw_data)
test_x = new_raw_data[10,25:36,70:81]
test_2 = new_raw_data[10]
plt.imshow(test_x,cmap = 'gray')
plt.show()   

#train_data
train_X = []
train_Y = []
input_train_size = 25
half_train_size = int (input_train_size / 2)
for i in range(20):
    index = poumain_list[i + 2]
    print(index)
    deal_poumain_x = new_raw_data[index]
    deal_poumain_y = data_label[i + 2]
    for j in range(half_train_size,(time_size - half_train_size)):
        for k in range(half_train_size,(cross_size - half_train_size )):
            batch_x = np.zeros((input_train_size,input_train_size))
            batch_x = new_raw_data[index,j - half_train_size : j + half_train_size + 1, k - half_train_size : k + half_train_size + 1]
            train_X.append(batch_x)
            train_Y.append(deal_poumain_y[j][k])
print (train_Y)
train_x = np.array(train_X)
train_y = np.array(train_Y)


np.save("train_x.npy",train_x)
np.save("train_y.npy",train_y)
np.save("data_label.npy",data_label)
np.save("data.npy",new_raw_data)
print('over')

 
