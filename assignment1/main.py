import numpy as np
import random as rd
from datetime import datetime
import pandas as pd 
from statistics import mean 
import random as rd
# from scipy.stats import mode


# current date and time
now = datetime.now()

import math
filepath = 'C:/Users/Ninja/Work/4 year 1 term/Computer Intelligence/neural_networks/assignment1/data.csv'
input_data =[]
output_data =[]
output_data_TEST =[]
gradient_layer_hid =[]
gradient_end = []
int_node = []
bias = []
hid_node = []
out_node = []
layer_1 = []
layer_hid = []
layer_end = []
v_struc =[] 
model_src = []
model_src2 = []
w_t_1_ = []
cnt = 0
TP = []
TN = []
FP = []
FN = []

with open(filepath) as fp:
    
    line = fp.readline()
    # list_ar =[]
    while line:
        tmp_split = line.split()

        input_data.append(tmp_split[0:23])
        output_data.append(tmp_split[23])

        line = fp.readline()
        cnt+=1
input_attri = pd.DataFrame(input_data,columns = ["surgery","Age","Hospital Number","rectal temperature","pulse","respiratory rate","temperature of extremities","peripheral pulse","mucous membranes","capillary refill time","pain","peristalsis","abdominal distension","nasogastric tube","nasogastric reflux","nasogastric reflux PH","rectal examination","abdomen","packed cell volume","total protein","abdominocentesis appearance","abdomcentesis total protein","outcome"])
# output_attri = output_data

# print(input_attri)
for i in range(30):
    # input_attri.index
    print(input_attri[rd.randint(0,300)])
# input_attri.to_csv (r'assignment1/data.csv', index = False, header=True)
       

def see_layer():
    print("Node Input:",int_node)
    print("layer_frist:")
    for  i in range(len(layer_1)):
        print(layer_1[i])
    print("Bias :", bias)
    for i in range(len(hid_node)):
        if(i>=1):
            print("layer_hidden :",i-1)
            for  j in range(len(layer_hid[i-1])):
                print(layer_hid[i-1][j])
        print("Node Hidden:[",i,"] :",hid_node[i])
    print("layer_end:")
    for  i in range(len(layer_end)):
        print(layer_end[i])
    print("Node Output:",out_node)
    
    return 0
def make_input_node(in_node):
    inn = []
    for i in range(in_node):
        inn.append(0)
    return inn
def make_hidden_node(hid_node = [] ):
    hid = []
    for i in range(hid_node[0]):
        layer = []
        for j in range(hid_node[i+1]):
            layer.append(0)
        hid.append(layer)
        # v_struc.append(layer)
    return hid
def make_output_node(out_node):
    out = []
    for i in range(out_node):
        out.append(0)
    # v_struc.append(out)
    return out
def define_weight(left = [],right = []):
    wg = []
    for i in range(len(left)):
        tmp_wg = []
        for j in range(len(right)):
            tmp_wg.append(rd.random())
        wg.append(tmp_wg)
    # w_t_1_.append(wg)
    return wg
def make_structure_NN(_input=0,_hidden =[],_output = 0):
    _in = make_input_node(_input)
    _bi = random_Value(_hidden[1])
    _hi = make_hidden_node(_hidden)
    _ot = make_output_node(_output)
    v_struc.append(make_input_node(_input))
    v_struc.append(make_hidden_node(_hidden))
    v_struc.append(make_output_node(_output))
    layer_1 = define_weight(make_input_node(_input),make_hidden_node(_hidden)[0])
    w_t_1_.append(define_weight(make_input_node(_input),make_hidden_node(_hidden)[0]))
    tmp = []
    w_t_1_.append(tmp)
    for i in range(len(_hidden)-2):
        tmp_layer = define_weight(make_hidden_node(_hidden)[i],make_hidden_node(_hidden)[i+1])
        layer_hid.append(tmp_layer)
        w_t_1_[1].append( define_weight(make_hidden_node(_hidden)[i],make_hidden_node(_hidden)[i+1]))
    layer_end = define_weight(make_hidden_node(_hidden)[len(_hidden)-2],make_output_node(_output))   
    w_t_1_.append(define_weight(make_hidden_node(_hidden)[len(_hidden)-2],make_output_node(_output)))
    return layer_1,layer_hid,layer_end,_in,_hi,_ot,_bi
def random_Value(lenght = 0):
    list_tmp = []
    for i in range(lenght):
        list_tmp.append(rd.random())
    return list_tmp

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def der_sigmoid(x):
    return x * (1-x)

def To_list_int(tmp_list =[]):
    for  i in range(len(tmp_list)):
        tmp_list[i] =  int(tmp_list[i])
    return tmp_list
def int_to_hid(_input = []):
    for i  in range(len(hid_node[0])):
        sum = 0.0
        for j in range(len(_input)):
            sum += (float(_input[j])*float(layer_1[j][i]))
        hid_node[0][i] = sigmoid(sum)
        v_struc[1][0][i] = sum
    return 0

def hid_to_hid(hid =[],num = 0):
    for i in range(len(hid[0])):
        sum = 0.0
        for j in range(len(hid_node[num-1])):
            sum += (float(hid_node[num-1][j])*float(hid[j][i]))
        hid_node[num][i] = sigmoid(sum)
        v_struc[1][num][i] = sum
    return 0

def hid_to_out():
    for i in range(len(out_node)):
        sum = 0.0
        for j in range(len(hid_node[len(hid_node)-1])):
            
            sum += (float(hid_node[len(hid_node)-1][j]) * float(layer_end[j][i]))
            # print("layer_w",j,i,"",hid_node[len(hid_node)-1][j],layer_end[j][i],sum)
        out_node[i] = sigmoid(sum)
        # print("v",v_struc[2])
        v_struc[2][i] = sum
        # print("[",2,i,"]",sum)
        
    return 0
    
def forwardComputation(tmp_in = [],tmp_hid = [],tmp_out = []):
    int_to_hid(tmp_in)
    for i in range(tmp_hid[0]-1):
        hid_to_hid(layer_hid[i],i+1) #[layer number,hidden_node number]
    hid_to_out()
    return 0 

def hid_to_hid_b(layer_j = [] ,layer_i =[],g_layer_k = [],index = 0,l = 0,m = 0,ck = 0):
    
    gradient_hid =[]
    for j in range(len(layer_j)):
        sum_gradient_layerK = 0.0
        sum_w = 0.0
        g = 0.0 
        for k in range(len(g_layer_k)):
           
            sum_gradient_layerK += g_layer_k[k]
            sum_w += layer_j[j][k]
        g = der_sigmoid(hid_node[index+1][j])*sum_gradient_layerK*sum_w
        for i in range(len(layer_i)):
            y = hid_node[index][i]
            del_w = m*(layer_hid[index][i][j] - w_t_1_[1][index][i][j]) + (l*(g*y))
            w_t_1_[1][index][i][j] = layer_hid[index][i][j]
            layer_hid[index][i][j] = layer_hid[index][i][j] + del_w
        gradient_hid.append(g)

    return gradient_hid 

def hid_to_int(layer_j = [] ,layer_i =[],g_layer_k = [],index = 0,l = 0,m = 0,ck = 0):
    
    for j in range(len(layer_j)):
        sum_gradient_layerK = 0.0
        sum_w = 0.0
        g = 0.0
        for k in range(len(layer_j[j])):
           
            sum_gradient_layerK += g_layer_k[k]
            sum_w += layer_j[j][k]
        g = (der_sigmoid(hid_node[0][j])*sum_gradient_layerK)*sum_w
        for i in range(len(layer_i)):  
            y = v_struc[0][i]
            del_w = m*(layer_1[i][j] - w_t_1_[0][i][j]) + (l*(g*y))
            w_t_1_[0][i][j] = layer_1[i][j]     
            layer_1[i][j] = layer_1[i][j] + del_w
    return 0

def out_to_hid(o =[],e = [],index_hid = 0,l = 0,m = 0):
    gradient_end = []
    g = 0 
    for j in range(len(o)):
        g = e[j]*der_sigmoid(out_node[j])
        for i in range(len(layer_end)):
            # print("W_t-1:",w_t_1_[2][i][j])
            # print("W_t  :",layer_end[i][j])
            y = hid_node[index_hid][i]
            del_w = m*(layer_end[i][j] - w_t_1_[2][i][j]) + (l*(g*y))
            # print("layer_end[",i,"][",j,"]", layer_end[i][j])
            w_t_1_[2][i][j] = layer_end[i][j]
            layer_end[i][j] = layer_end[i][j] + del_w
            
        gradient_end.append(g)

    return gradient_end
    
def backwardComputation(tmp_in = [],tmp_hid = [],tmp_out = [],e =[],lr = 0 ,mr =0):
    gradient_layer_hid = out_to_hid(tmp_out,e,tmp_hid[0]-1,lr,mr)
    check = 0    
    for i in range(tmp_hid[0]-2,-1,-1):
        if(i >= tmp_hid[0]-2): #local
            gradient_layer_hid = hid_to_hid_b(layer_end,layer_hid[i],gradient_layer_hid,i,lr,mr,check)
        else:
            gradient_layer_hid = hid_to_hid_b(layer_hid[i+1],layer_hid[i],gradient_layer_hid,i,lr,mr,check)
   
    hid_to_int(layer_hid[0],layer_1,gradient_layer_hid,0,lr,mr,check)
    return 0 

def error_of_result(d =[],o =[]):
    e  = []
    for j in range(len(d)):
        e.append(0)
    # print("whatt",d)
    sum_error = 0.0
    for i  in range(len(d)):
        e[i] = (d[i] - o[i])
        # print("error^2",np.power(e[i],2))
        sum_error += np.power(e[i],2)
    return e,sum_error/2.0

def Normalizing_of_Input(o = []):
    v_max=max(o)
    v_min=min(o)
    for i in range(len(o)):
        o[i] = (o[i]-v_min)/(v_max-v_min)
    return o
# main() 
def show(count= 0, finish = 0):
    show_output = (count/finish)*50 
    num =0
    if(count%6 == 0 and show_output <= 51):
        print("[", end ='')
        for x in range(int(show_output+1)):
            num +=1
            print("#", end ='')
        if(num != 50):
            for y in range(50-num):
                 print("-", end ='')
        print("]",num*2,"%")
        print("\n")

def clear_value_2D( v =[]):
    for i in range(len(v)):
        for j in range(len(v[i])):
            v[i][j] = 0
    return v
def clear_value_3D( v =[]):
    for i in range(len(v)):       
        for j in range(len(v[i])):
            for k in range(len(v[i][j])):
                v[i][j][k] = 0
    return v

def select_input(input_ = 0 ,training = 0, test = 0):
    round = 100 / test
    num_train = int(input_*(training/100))
    num_test = input_ - num_train
    print("-----------------------------------------")
    print("n :",input_)
    print("n_train",num_train)
    print("n_test",num_test)
    print("-----------------------------------------")
    for z in range(int(round)):
        ar_train = []
        ar_test = []
        tmp = []
        tmp2 = []
        begin_test = 0
        for x in range(num_train):
            x = x+(num_test*z)
            ar_train.append(x%input_)
            begin_test = x + 1
        # tmp.append(ar_train)
        for y in range(num_test):
            if(z >= 1):
                y = y+(num_test*z) - num_test
            else:
                y = y +(num_test*z)
            ar_test.append((begin_test+y)%input_)
        # tmp2.append(ar_test)
        model_src.append(ar_train)
        model_src2.append(ar_test)
    # print(model_src[1],"lol")
    # print(model_src2[1],"lol")
def select_model_best():
    sum_error_avg = 0
    N_ = 0
    index = 0
    tmp1 = 0
    tmp2 = 0
    max_index = 0
    for z in range(10):
        print("Model[",z,"]")   
        # len(model_src[z][0])
        int_node = []
        for count in range(len(model_src[z])):
            v_struc[0] = []
            v_struc[1] = clear_value_2D(v_struc[1])
            v_struc[2] = [0,0,0]

            # data = Normalizing_of_Input(To_list_int(input_data[model_src[z][0][count]]))
            int_node = input_data[model_src[z][count]]
            # print(int_node,"opss")
            v_struc[0] = int_node
            

            
            forwardComputation(int_node,str_hid,out_node)
           
            error,sum_e = error_of_result(output_data[model_src[z][count]],out_node)
          
            backwardComputation(int_node,str_hid,out_node,error,learn_rate,momentum_rate)
            # print(out_node[0],output_data[model_src[z][count]][0])
            # print(out_node[1],output_data[model_src[z][count]][1])
            sum_error_avg += sum_e
            N_ = count + 1

        print("--------------------",z)
        sum_error_avg = sum_error_avg / N_
        tmp1 = sum_error_avg
        print("sum_error_av",sum_error_avg)

        if(tmp1 < tmp2 ):
            # print("tmp1,tmp2",tmp1,tmp2)
            tmp2 = tmp1
            max_index = z
        if(z<=0):
            tmp2 = tmp1

    return max_index
def mode(data):  
    modecnt=0
#for count of number appearing
    for i in range(len(data)):
        icount=data.count(data[i])
#for storing count of each number in list will be stored
        if icount>modecnt:
#the loop activates if current count if greater than the previous count 
            mode=data[i]
#here the mode of number is stored 
            modecnt=icount
#count of the appearance of number is stored
    return mode
# print mode(data1)

def to_float( o = []):
    get_value = []
    for i in range(len(o)):
        if(o[i] != "?"):
            get_value.append(float(o[i]))

    mean_list =  mean(get_value)
    for i in range(len(o)):
        if(o[i] == "?"):
            o[i] = mean_list    
        else:
            o[i] = float(o[i])
    return o


output = 3 
show_output_num = 0
str_hid = [3,4,8,4] #hidden_structure [จำนวนlayer,node1,node2,...]
layer_1,layer_hid,layer_end,int_node,hid_node,out_node,bias = make_structure_NN(21,str_hid,output)
learn_rate = 12.0
momentum_rate = 0.8
old_w_local = layer_end # แก้ w รอบก่อนนนั้น

select_input(len(input_data),80,20)
sum_error_avg = 0
N_ = 1
epoch = 500

w_t_1_[0] = clear_value_2D(w_t_1_[0])
w_t_1_[1] = clear_value_3D(w_t_1_[1])
w_t_1_[2] = clear_value_2D(w_t_1_[2])

columns_str = ["surgery","Age","rectal temperature","pulse","respiratory rate","temperature of extremities","peripheral pulse","mucous membranes","capillary refill time","pain","peristalsis","abdominal distension","nasogastric tube","nasogastric reflux","nasogastric reflux PH","rectal examination","abdomen","packed cell volume","total protein","abdominocentesis appearance","abdomcentesis total protein"]
input_attri = input_attri[columns_str]

table_columns = []
for  i in range(len(columns_str)):
    g = input_attri[columns_str[i]].tolist()
    g = to_float(g)
    # print(g)
    normal_ = Normalizing_of_Input(g)
    # print(normal_)
    table_columns.append(normal_)
table_columns = pd.DataFrame(table_columns)

# print(table_columns[0].tolist())

# print(output_attri)
out_desir  = []
for i in range(len(output_attri)):
    g = []
    if(output_attri[i] == "1"):
        g = [1,0,0]
    elif(output_attri[i] == "2"):
        g = [0,1,0]
    elif(output_attri[i] == "3"):
        g = [0,0,1]
    out_desir.append(g)

# print(out_desir[0])
# print(pd.DataFrame(v_struc))
print(pd.DataFrame(model_src))
# z = select_model_best()
z = 0 
array_SumErrorAvg = []
see_layer()
print("-----------------------------------------")
# for i in range(10):
#     sum_error_avg = 0
#     print("train_data [",i + 1,"]" )
    # sum_error_avg = 0 
# print(out_node[0],output_data[model_src2[z][count]][0])
# print(out_node[1],output_data[model_src2[z][count]][1])

# train #####################################################
for h in range(epoch):
    print("epoch :",h)
    sum_error_avg = 0 
    # len(model_src[z][0])
    for count in range(len(model_src[z])):
        # print("count :",count)
        v_struc[0] = []
        v_struc[1] = clear_value_2D(v_struc[1])
        v_struc[2] = [0,0,0]
       
        # print(out_node[0],output_data[model_src[z][count]][0])
        # print(out_node[1],output_data[model_src[z][count]][1])
        # data = Normalizing_of_Input(To_list_int(input_data[model_src[z][0][count]]))
        # input_data[model_src[z][count]]
        int_node = table_columns[model_src[z][count]].tolist()
        
        v_struc[0] = int_node
        forwardComputation(int_node,str_hid,out_node)
        error,sum_e = error_of_result(out_desir[model_src[z][count]],out_node)
        # print("error:", error)
        backwardComputation(int_node,str_hid,out_node,error,learn_rate,momentum_rate)
        sum_error_avg += sum_e
        N_ = count + 1
    print("--------------------")
    sum_error_avg = sum_error_avg / N_
    print("sum_error_av",sum_error_avg)

print("---------------- test ---------------------\n")
sum_error_avg = 0

# print("layer_1",layer_1)
# print("layer_hid",layer_hid)
# print("layer_end",layer_end)
for count in range(len(model_src2[z])):
    print("test_data_[",count,"]","Data_index :",model_src2[z][count])
#     # data = Normalizing_of_Input(To_list_int(input_data[model_src[z][count]]))
    int_node = table_columns[model_src2[z][count]].tolist()

    forwardComputation(int_node,str_hid,out_node)
  
#     print(out_node[0],output_data[model_src2[z][count]][0])
#     print(out_node[1],output_data[model_src2[z][count]][1])

    if(out_desir[model_src2[z][count]][0] == 1):
        if(out_node[0] > 0.5):
            TP.append(out_node[0])
        elif(out_node[0] < 0.5) :
            FN.append(out_node[0])
    elif(out_desir[model_src2[z][count]][0] == 0):
        if(out_node[0] > 0.5):
            FP.append(out_node[0])
        elif(out_node[0] < 0.5) :
            TP.append(out_node[0])

    if(out_desir[model_src2[z][count]][1] == 1):
        if(out_node[1] > 0.5):
            TP.append(out_node[1])
        elif(out_node[1] < 0.5) :
            FN.append(out_node[1])
    elif(out_desir[model_src2[z][count]][1] == 0):
        if(out_node[1] > 0.5):
            FP.append(out_node[1])
        elif(out_node[1] < 0.5) :
            TP.append(out_node[1])
    

    if(out_desir[model_src2[z][count]][2] == 1):
        if(out_node[2] > 0.5):
            TP.append(out_node[2])
        elif(out_node[2] < 0.5) :
            FN.append(out_node[2])
    elif(out_desir[model_src2[z][count]][2] == 0):
        if(out_node[2] > 0.5):
            FP.append(out_node[2])
        elif(out_node[2] < 0.5) :
            TP.append(out_node[2])
   

    print("--------------------")
#     # print("error",e,"sum_error_avg",sum_error_avg)
print("TP",len(TP))
print("TN",len(TN))
print("FP",len(FP))
print("FN",len(FN))
print("Accuracy :",((len(TP)+len(TN))/(len(TP)+len(TN)+len(FP)+len(FN)))*100.0 ,"%")
# # sum_error_avg = sum_error_avg/N_
# # print("sum_error_avg",abs(sum_error_avg*100) , "%")
# now2 = datetime.now()
# timestamp2 = datetime.timestamp(now2)
# timestamp = datetime.timestamp(now)
# timestamp = datetime.fromtimestamp(timestamp)
# print("timestamp =", timestamp)
# timestamp = datetime.fromtimestamp(timestamp2)
# print("timestamp_finish =", timestamp)