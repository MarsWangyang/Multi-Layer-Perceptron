#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import random
import math

window = tk.Tk()
window.title('MLP')
window.geometry('700x700')

#ListBox
#select file
def file_selection():
    value = lb.get(lb.curselection())                        #滑鼠選到的檔案稱value
    var4.set(value)    
    raw_data(value)


def sig(v):
    if v >= 0.6:
        return 1
    elif v < 0.6:
        return 0
def sig3(v):
    if v < 0.3:
        return 0
    elif v >= 0.3 and v < 0.666:
        return 0.333
    else:
        return 1
    
#整理資料    
def raw_data(value):
    learning_rate_percent = float(var1.get())
    learning_rate = learning_rate_percent / 100
    iteration = int(var2.get())
        
    data = []
    with open( value ,'r', encoding = 'utf8' ) as f :
        for line in f:
            data.append([float(ele) for ele in line.split()])        
        small = data[0][2]                                            #先將第一筆資料的期望輸出叫做small
        for i in range(0 , len(data),1):
            if data[i][2] < small:
                small = data[i][2]                                    #檢查每筆資料的期望輸出有沒有會是最小
            elif data[i][2] > small:
                big = data[i][2]                                      #找到最大的鍵結值       
        for i in range(0 , len(data)):
            if data[i][2] == small:
                dmin = small
            elif data[i][2] == big:
                dmax = big                   
        for i in range(0 , len(data)):                                #利用將正規化整理期望輸出
            if (data[i][2] != small) or (data[i][2] != big):
                data[i][2] = round((data[i][2] - dmin) / (dmax - dmin),3)
            
        for i in range(0,len(data)):                                  #分類出三種期望輸出和兩種期望輸出的data，如果是三種輸出則:output_3 = 1
            if (data[i][2] != 0) and (data[i][2] != 1):
                output_3 = 1
                break
            else:
                output_3 = 0
   
        lim_bigx1 = data[0][0]     
        lim_smallx1 = data[1][0]
        lim_bigx2 = data[0][1]
        lim_smallx2 = data[1][1]
        
        for i in range(0,len(data)):
            if data[i][0] > lim_bigx1:
                lim_bigx1 = data[i][0]
            if data[i][0] < lim_smallx1:
                lim_smallx1 = data[i][0]
            if data[i][1] > lim_bigx2:
                lim_bigx2 = data[i][1]
            if data[i][1] < lim_smallx2:
                lim_smallx2 = data[i][1]
        
        random.shuffle(data)
        prepare(data,learning_rate,iteration,output_3,lim_bigx1,lim_smallx1,lim_bigx2,lim_smallx2)

def prepare(data,learning_rate,iteration,output_3,lim_bigx1,lim_smallx1,lim_bigx2,lim_smallx2):
    training_data = []
    testing_data = []
    for i in range(0 , int(len(data)*(2/3))):
        training_data.append(data[i])   
    for x in range(int(len(data)*(2/3)) , len(data)):
        testing_data.append(data[x])
    
    a = np.array(training_data , dtype = np.float)     #turn training data into array
    b = np.array(testing_data , dtype = np.float)
    c = np.full(len(training_data),-1)                 #add -1 items for X0
    d = np.full(len(testing_data),-1) 
    
    training_data_x = a[:,:2]
    training_data_y = a[:,2]
    testing_data_x = b[:,:2]
    testing_data_y = b[:,2]
    
    input_training_data = np.c_[c,training_data_x]
    input_testing_data = np.c_[d,testing_data_x]
    
    #weight_layer1
    w1 = np.random.uniform(-1,1,size = (4,2))
    e = np.full(4,-1)
    weight1 = np.c_[e,w1]
    
    #weight_layer2
    w2 = np.random.uniform(-1,1,size = (2,4))
    f = np.full(2,-1)
    weight2 = np.c_[f,w2]
    
    #weight_outputlayer
    wy = np.random.uniform(-1,1,size = (1,2))
    g = np.full(1,-1)
    weighty = np.c_[g,wy]
    
    x_T = []
    for i in range(0,len(input_training_data)):
        x_T.append(np.transpose(input_training_data[i]))
        
    x_Test = []
    for i in range(0,len(input_testing_data)):
        x_Test.append(np.transpose(input_testing_data[i]))
    
    train(data,weight1,weight2,weighty,x_T,training_data_x,training_data_y,learning_rate,output_3,input_testing_data,testing_data_x,testing_data_y,x_Test,iteration,lim_bigx1,lim_smallx1,lim_bigx2,lim_smallx2)

def sigmoid(v):
    return 1 / (1 + math.exp(-v))

#設計 2 hidden layers(1st:4個 2nd:2個)
def train(data,weight1,weight2,weighty,x_T,training_data_x,training_data_y,learning_rate,output_3,input_testing_data,testing_data_x,testing_data_y,x_Test,iteration,lim_bigx1,lim_smallx1,lim_bigx2,lim_smallx2): 
    counter = 1
    right = 0
    fail = 0
    training_accuracy = 0.00
    J = 1.000 #condition of convergence
    E = 0.000 #total平方差
#-----------------------------------------------訓練鍵結值
    while(counter < iteration and J > 0.00001):   
        for i in range(0,len(x_T)):  
            #前饋
            v = np.dot(weight1,x_T[i])                   #1st hidden layer的神經元v值

            y_layer1 = [-1]
            for x in range(0,4):
                y_layer1.append(sigmoid(v[x]))
            y_layer1_array = np.array(y_layer1)

            v2 = np.dot(weight2,y_layer1)

            y_layer2 = [-1]
            for k in range(0,2):
                y_layer2.append(sigmoid(v2[k]))
            y_layer2_array = np.array(y_layer2)

            v3 = np.dot(weighty,y_layer2)
            y_output = sigmoid(v3)

            #backpopagation level
            delta_output = (training_data_y[i] - y_output) * y_output * (1 - y_output)

            delta_layer2 = []
            for p in range(1,3):
                delta_layer2.append(y_layer2[p] * (1 - y_layer2[p]) * (delta_output * weighty[0][p]))

            delta_layer1 = []
            for j in range(1,5):
                delta_layer1.append(y_layer1[j] * (1 - y_layer1[j]) * (delta_layer2[0] * weight2[0][j] + delta_layer2[1] * weight2[1][j]))
            
            #調整鍵結值
            weighty = weighty + learning_rate * delta_output * y_layer2_array
            weight2[0] = weight2[0] + learning_rate * delta_layer2[0] * y_layer1_array 
            weight2[1] = weight2[1] + learning_rate * delta_layer2[1] * y_layer1_array
            weight1[0] = weight1[0] + learning_rate * delta_layer1[0] * x_T[i][0]
            weight1[1] = weight1[1] + learning_rate * delta_layer1[1] * x_T[i][1]
            weight1[2] = weight1[2] + learning_rate * delta_layer1[2] * x_T[i][2]
            E = E + 0.5 * pow((training_data_y[i] - y_output), 2)
            
#----------------------------------------------------------------------------------            
            if output_3 == 1 :
                y_acc = sig3(y_output)
                if(y_acc != training_data_y[i]):
                    fail = fail + 1 
                    
                else:
                    right = right + 1
            else :
                y_acc = sig(y_output)
                if(y_acc != training_data_y[i]):
                    fail = fail + 1
                else:
                    right = right + 1      
            
            training_accuracy = (right / (right + fail)) * 100 
        #---------------學習率調整---------------------
        N = learning_rate / (1 + counter / iteration)
        N_p = learning_rate
        J = N_p - N
        learning_rate = N
        #---------------RMSE--------------------------
        counter += 1
    RMSE = round(math.sqrt(E / counter) , 7)
    var5.set(training_accuracy)
    var8.set(RMSE)
    
    #--------------------drawing----------------------
    plt.figure()
    plt.subplot(2,1,1)
    if output_3 == 0:
        for i in range(0,len(data)):
            if data[i][2] == 1:
                data_marker = "o"
                data_color = 'b'
            else:
                data_marker = "x"
                data_color = 'r'
            plt.plot(data[i][0],data[i][1], c = data_color , marker = data_marker)
    else:
        for i in range(0,len(data)):
            if data[i][2] == 1:
                data_marker = "o"
                data_color = 'b'
            elif data[i][2] == 0.333:
                data_marker = "x"
                data_color = 'g'
            else:
                data_marker = "^"
                data_color ='r'
            plt.plot(data[i][0],data[i][1], c = data_color , marker = data_marker)
    
    plt.title("MLP raw data")
    plt.xlim(lim_smallx1-1 , lim_bigx1+1)
    plt.ylim(lim_smallx2-1 , lim_bigx2+1)
    plt.xlabel('X1')
    plt.ylabel('X2')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['left'].set_position(('data',0))
    
    test(data,weight1,weight2,weighty,input_testing_data,testing_data_x,testing_data_y,learning_rate,x_Test,output_3,lim_bigx1,lim_smallx1,lim_bigx2,lim_smallx2)

 
#test        
def test(data,weight1,weight2,weighty,input_testing_data,testing_data_x,testing_data_y,learning_rate,x_Test,output_3,lim_bigx1,lim_smallx1,lim_bigx2,lim_smallx2):
    right = 0
    fail = 0
    testing_accuracy = 0.00
    test_y = [] 
#--------------------------testing-----------------------------------------------
    for i in range(0,len(x_Test)):
        vlayer1 = np.dot(weight1,x_Test[i])

        ylayer1 = [-1]
        for x in range(0,4):
            ylayer1.append(sigmoid(vlayer1[x]))
        ylayer1_array = np.array(ylayer1)

        vlayer2 = np.dot(weight2,ylayer1)

        ylayer2 = [-1]
        for k in range(0,2):
            ylayer2.append(sigmoid(vlayer2[k]))
        ylayer2_array = np.array(ylayer2)

        vlayer3 = np.dot(weighty,ylayer2)
        youtput = sigmoid(vlayer3) 
#------------------------------------------------------------------------                          
        if output_3 == 1 :
            y_acc = sig3(youtput)
            if(y_acc != testing_data_y[i]):
                fail = fail + 1 
            else:
                right = right + 1
        else :
            y_acc = sig(youtput)
            if(y_acc != testing_data_y[i]):
                fail = fail + 1
            else:
                right = right + 1  
        test_y.append(y_acc)
        
        testing_accuracy = (right / (right + fail)) * 100  
    
    var6.set(testing_accuracy)
    var7.set(str(weight1[0]))
    var9.set(str(weight1[1]))
    var10.set(str(weight1[2]))
    var11.set(str(weight1[3]))
    var12.set(str(weight2[0]))
    var13.set(str(weight2[1]))
    var14.set(str(weighty))
#-----------------drawing-----------------------
    plt.subplot(2,1,2)
    if output_3 == 0:
        for i in range(0,len(testing_data_x)):
            if test_y[i] == 1:
                data_marker = "o"
                data_color = 'b'
            else:
                data_marker = "x"
                data_color = 'r'
            plt.plot(testing_data_x[i][0],testing_data_x[i][1], c = data_color , marker = data_marker)
    else:
        for i in range(0,len(testing_data_x)):
            if test_y[i] == 1:
                data_marker = "o"
                data_color = 'b'
            elif test_y[i] == 0.333:
                data_marker = "x"
                data_color = 'g'
            else:
                data_marker = "^"
                data_color ='r'
            plt.plot(testing_data_x[i][0],testing_data_x[i][1], c = data_color , marker = data_marker)
        
    plt.title("MLP Test Result")
    plt.xlim(lim_smallx1-1 , lim_bigx1+1)
    plt.ylim(lim_smallx2-1 , lim_bigx2+1)
    plt.xlabel('X1')
    plt.ylabel('X2')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['left'].set_position(('data',0))
    plt.show()
 
#GUI
#Label&Entry
var1 = tk.StringVar()
L1 = tk.Label(window,text = '學習率learning rate(%) = ').place(x = 120, y = 310)
entry_learning = tk.Entry(window,textvariable = var1)
entry_learning.place(x = 300, y = 310 )

var2 = tk.StringVar()
L2 = tk.Label(window,text = '疊代次數iteration(次)    = ').place(x = 120, y = 360)
entry_iteration = tk.Entry(window,textvariable = var2)
entry_iteration.place(x = 300 , y = 360)

var4 = tk.StringVar()
L4 = tk.Label(window,text = '您所選擇的檔案 ： ').place(x = 330, y = 135 )
entry_choose = tk.Entry(window,textvariable = var4)
entry_choose.place(x = 330 , y = 165)
    
L_input = tk.Label(window, text = '請選擇您要訓練的檔案：')
L_input.place(x = 120 , y = 50)
var_data = tk.StringVar()
var_data.set(('perceptron1.txt','perceptron2.txt','2Ccircle1.txt','2Circle1.txt','2Circle2.txt','2CloseS.txt','2CloseS2.txt','2CloseS3.txt',
            '2cring.txt','2CS.txt','2Hcircle1.txt','2ring.txt'))
lb = tk.Listbox(window, listvariable = var_data, height = 12)
lb.place(x = 120 , y = 70)

var5 = tk.StringVar()
L5 = tk.Label(window,text = ' 訓練辨識率training accuracy(%) = ').place(x = 120 , y = 450)
entry_5  = tk.Entry(window,textvariable = var5 )
entry_5.place(x = 320 , y = 450)

var6 = tk.StringVar()
L6 = tk.Label(window,text = ' 測試辨識率testing accuracy(%)  = ').place(x = 120 , y = 500)
entry_6  = tk.Entry(window,textvariable = var6 )
entry_6.place(x = 320 , y = 500)

var7 = tk.StringVar()  
L7 = tk.Label(window,text = ' 鍵結值Synaptic Weights              =            第一隱藏層                   第二隱藏層                        輸出層').place(x = 120 , y = 550)
entry_1_1  = tk.Entry(window,textvariable = var7 , width = 15  )
entry_1_1.place(x = 320 , y = 570)
var9 = tk.StringVar()
entry_1_2  = tk.Entry(window,textvariable = var9 , width = 15  )
entry_1_2.place(x = 320 , y = 590)
var10 = tk.StringVar()
entry_1_3  = tk.Entry(window,textvariable = var10 , width = 15  )
entry_1_3.place(x = 320 , y = 610)
var11 = tk.StringVar()
entry_1_4  = tk.Entry(window,textvariable = var11 , width = 15  )
entry_1_4.place(x = 320 , y = 630)
var12 = tk.StringVar()
entry_2_1  = tk.Entry(window,textvariable = var12 , width = 15  )
entry_2_1.place(x = 440 , y = 570)
var13 = tk.StringVar()
entry_2_2  = tk.Entry(window,textvariable = var13 , width = 15  )
entry_2_2.place(x = 440 , y = 590)
var14 = tk.StringVar()
entry_3_1  = tk.Entry(window,textvariable = var14 , width = 15  )
entry_3_1.place(x = 560 , y = 570)

var15 = tk.StringVar()
L15 = tk.Label(window,text = ' 第一個神經元 ').place(x = 230 , y = 570)
var16 = tk.StringVar()
L16 = tk.Label(window,text = ' 第二個神經元 ').place(x = 230 , y = 590)
var17 = tk.StringVar()
L17 = tk.Label(window,text = ' 第三個神經元 ').place(x = 230 , y = 610)
var18 = tk.StringVar()
L18 = tk.Label(window,text = ' 第四個神經元 ').place(x = 230 , y = 630)



var8 = tk.StringVar()
L8 = tk.Label(window,text = ' 均方根誤差RMSE                           =').place(x = 120 , y = 660)
entry_8  = tk.Entry(window,textvariable = var8 )
entry_8.place(x = 320 , y = 660)

L_result = tk.Label(window , text = ' 請輸入您所想要的數值 : ')
L_result.place(x = 20 , y = 285)

L_result = tk.Label(window , text = ' 訓練結果 : ')
L_result.place(x = 60 , y = 420)

#button
b_go = tk.Button(window,text = 'Go' ,bg = 'green' , width = 20 ,height = 7 , command= file_selection )
b_go.place(x = 530 , y = 280)

window.mainloop()        

