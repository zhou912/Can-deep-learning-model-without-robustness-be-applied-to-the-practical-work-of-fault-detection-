# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 18:45:41 2018
TODO:注意此处有bug，最后一个剖面的label没有读入进去，没有保存最后一次不变的循环
暂时处理方式为 在fault_inline最后一行随机添加一个剖面信息,然后读入的poumain_list减1
"""
import numpy as np
import scipy.io as sio
from collections import Counter
np.set_printoptions(threshold = np.inf)
def importfile(path, name):
    dictname = name.replace('.mat','')
    rawdata = sio.loadmat(path + '\\' +  name)[dictname]
    return rawdata

'''
读取线段坐标信息
'''
def getLines(path, filename):
    fr = open(path + '/' + filename)
    arrayOfLines = fr.readlines()
    arrayOfLines = arrayOfLines[1:]
    Lines = []
    for line in arrayOfLines:
        line = line.strip().split()
        line[1] = int(line[1])
        line[2] = int(line[2])
        line[3] = int(float(line[3])/2)*2
        Lines.append(line)
    fr.close()
    return Lines

'''
获得所有标记点的坐标
AddLines:[inline,xline,t]
'''
def getAddLines(Lines):
    dt = 2
    numberOfLines = len(Lines)
    AddLines = []
    for i in range(numberOfLines):
        if(1400<=Lines[i][2] and Lines[i][2]<=1800 and 800<=Lines[i][3] and Lines[i][3]<=1000 and 2000<=Lines[i][1] and Lines[i][1]<=2400):
            templine = Lines[i]
            if(templine[4] == '1' or templine[4] == '2'):
                Addt = templine[3]     
                Nextt = Lines[i+1][3]
                while(Addt < Nextt):
                    templine[3] = Addt
                    x = templine.copy() #如果直接append:后面的templine改变，list前面的也会改变，因为地址相同
                    AddLines.append(x[1:4])
#                    del x
                    Addt = Addt + dt
                while(Addt > Nextt):
                    templine[3] = Addt
                    x = templine.copy() #如果直接append:后面的templine改变，list前面的也会改变，因为地址相同,没有del的copy也不会增加内存爆炸,copy就是为了减少内存的使用
                    AddLines.append(x[1:4])
#                    del x
                    Addt = Addt - dt               
            else:
                AddLines.append(Lines[i][1:4])    
    return AddLines


'''
将数据体每个方向以复制方式扩充Half_size
'''
def dataEnlarge(seisdata, Half_size):
    size = seisdata.shape
    LargeData = np.zeros([size[0]+2*Half_size,size[1]+2*Half_size,size[2]+2*Half_size])
    LargeData[Half_size:size[0]+Half_size,Half_size:size[1]+Half_size,Half_size:size[2]+Half_size] = seisdata
    for i in range(size[0]):                    
        for k in range(Half_size):   
            LargeData[i+Half_size,k,:] = LargeData[i+Half_size,Half_size,:]
            LargeData[i+Half_size,size[1]+Half_size+k,:] = LargeData[i+Half_size,size[1]+Half_size-1,:]
        for k in range(Half_size):   
            LargeData[i+Half_size,:,k] = LargeData[i+Half_size,:,Half_size]             
            LargeData[i+Half_size,:,size[2]+Half_size+k] = LargeData[i+Half_size,:,size[2]+Half_size-1]           
    for k in range(Half_size): 
        LargeData[k,:,:] = LargeData[Half_size,:,:]
        LargeData[size[0]+Half_size+k,:,:] = LargeData[size[0]+Half_size-1,:,:]
    return LargeData 
            
            
    
'''
Shift_size:[,,]是为t,x,y方向移动的距离,归一起点为0
pos:[t,x,y]
'''
def dataMake(AddLines,Shift_size):
    data = []
    for dot in AddLines:
#        没有移动,是为方块左上角的点
        y = dot[0] - Shift_size[0]
        x = dot[1] - Shift_size[1]
        t = int((dot[2] - Shift_size[2])/2)
        test = [y,t,x]
        data.append(test)
    #print (data)
    output_data = []
    poumian_num = 0
    poumain_data = np.zeros((101, 401))
    poumain_list = []
    poumain_list.append(poumian_num)
    for dot in data:
        #print(dot)
        if dot[0] == poumian_num:
            poumain_data[dot[1],dot[2]]=1
        else:
            output_data.append(poumain_data)
            poumain_data = np.zeros((101,401))
            poumian_num = dot[0]
            poumain_list.append(poumian_num)
    return output_data,poumain_list



