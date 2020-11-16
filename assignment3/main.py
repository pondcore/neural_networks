import matplotlib.pyplot as plt
import numpy as np
import random as rd
import pandas as pd
import math

# defind variable
Genararion_number = 75
v_struc =[] 
w_t_1_ = []
layer_1 = []
output_data =[]
out_node = []
hid_node = []
layer_hid = []
layer_end = []


# funtion
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

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def random_Value(lenght = 0):
    list_tmp = []
    for i in range(lenght):
        list_tmp.append(rd.random())
    return list_tmp

def select_input(input_ = 0 ,training = 0, test = 0):
    round = 100 / test
    model_train = []
    model_test = []
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
        model_train.append(ar_train)
        model_test.append(ar_test)

    return model_train,model_test

def make_structure_NN(_input=0,_hidden =[],_output = 0):
    _in = make_input_node(_input)
    _bi = random_Value(_hidden[1])
    _hi = make_hidden_node(_hidden)
    _ot = make_output_node(_output)
    v_struc.append(make_input_node(_input))
    v_struc.append(make_hidden_node(_hidden))
    v_struc.append(make_output_node(_output))
    layer_1 = define_weight(make_input_node(_input),make_hidden_node(_hidden)[0])

    for i in range(len(_hidden)-2):
        tmp_layer = define_weight(make_hidden_node(_hidden)[i],make_hidden_node(_hidden)[i+1])
        layer_hid.append(tmp_layer)
    layer_end = define_weight(make_hidden_node(_hidden)[len(_hidden)-2],make_output_node(_output)) 

    return layer_1,layer_hid,layer_end,_in,_hi,_ot,_bi

def int_to_hid(_input = [],h_node = [],layer_1 = []):
    # print("H :", h_node)
    # print("layer_1 :",layer_1)
    # print("in :",len(_input))
    for i  in range(len(h_node[0])):
        sum = 0.0
        for j in range(len(_input)):
            sum += (float(_input[j])*float(layer_1[j][i]))
        h_node[0][i] = sigmoid(sum)
        # v_struc[1][0][i] = sum
    # print("H :", h_node)
    return h_node

def hid_to_hid(hid =[],num = 0, hid_node = []):
    for i in range(len(hid[0])):
        sum = 0.0
        for j in range(len(hid_node[num-1])):
            sum += (float(hid_node[num-1][j])*float(hid[j][i]))
        hid_node[num][i] = sigmoid(sum)
        # v_struc[1][num][i] = sum
    # print("H :",hid_node)
    return hid_node

def hid_to_out(out_node =[],hid_node = [],layer_end =[]):
    for i in range(len(out_node)):
        sum = 0.0
        for j in range(len(hid_node[len(hid_node)-1])):
            sum += (float(hid_node[len(hid_node)-1][j]) * float(layer_end[j][i]))
        out_node[i] = sigmoid(sum)
        # v_struc[2][i] = sum
    return out_node

def Normalizing_of_Input(o = []):
    for i in range(len(o)):
        o[i] = float(o[i])

    v_max=max(o)
    v_min=min(o)
    for i in range(len(o)):
        o[i] = (o[i]-v_min)/(v_max-v_min)
    return o

def forwardComputation(tmp_in = [],tmp_hid = [],tmp_out = [],int_node = [],hid_node = [],out_node = [],layer_1 = [],layer_hid= [],layer_end= []):
    # print("hidden ", hid_node)
    hid_node = int_to_hid(tmp_in,hid_node,layer_1)
    for i in range(tmp_hid[0]-1):
        hid_node = hid_to_hid(layer_hid[i],i+1,hid_node) #[layer number,hidden_node number]
    out_node = hid_to_out(out_node,hid_node,layer_end)
   
    return int_node,hid_node,out_node,layer_1,layer_hid,layer_end

def Initial_Popultion(N = 0 , data_ = []):
    return 0 
def Create_Offsring(Fit_A = [],struc_A = [],Fit_B = [],struc_B = []):
    select_node = int(rd.randrange(2,8))
    arr_A = []
    arr_B = []
    print("select_node_A :",select_node)
    # print("Fit_A :",Fit_A)
    if(select_node >= 2 and select_node < 5):
        # print("str_A :",struc_A[4][0])
        # print("str_A :",struc_A[4][0][select_node-2])
        for i in range(len(struc_A[0])):
            arr_A.append(struc_A[0][i][select_node-2])
    elif (select_node >= 5 and select_node < 8 ):
        # print("str_A :",struc_A[4][1])
        # print("str_A :",struc_A[4][1][select_node-5])
        
        for i in range(len(struc_A[1][0])):
            arr_A.append(struc_A[1][0][i][select_node-5])

    elif select_node >=8 :
        # print("str_A :",struc_A[5])
        # print("str_A :",struc_A[5][select_node-7])
        for i in range(len(struc_A[2])):
            arr_A.append(struc_A[2][i][select_node-7])

    # print("Fit_B :",Fit_B)
    if(select_node >= 2 and select_node < 5):
        # print("str_B :",struc_B[4][0])
        # print("str_B :",struc_B[4][0][select_node-2])
        for i in range(len(struc_B[0])):
            arr_B.append(struc_B[0][i][select_node-2])
    elif (select_node >= 5 and select_node < 8 ):
        # print("str_B :",struc_B[4][1])
        # print("str_B :",struc_B[4][1][select_node-5])
        for i in range(len(struc_B[1][0])):
            arr_B.append(struc_B[1][0][i][select_node-5])
    elif select_node >=8 :
        # print("str_B :",struc_B[5])
        # print("str_B :",struc_B[5][select_node-7])
        for i in range(len(struc_B[2])):
            arr_B.append(struc_B[2][i][select_node-7])
    
    #crossover
    if(select_node >= 2 and select_node < 5):
        for i in range(len(struc_B[0])):
            struc_B[0][i][select_node-2] = arr_A[i]
            struc_A[0][i][select_node-2] = arr_B[i]

    elif (select_node >= 5 and select_node < 8 ):
        for i in range(len(struc_B[1][0])):
            struc_B[1][0][i][select_node-5] = arr_A[i]
            struc_A[1][0][i][select_node-5] = arr_B[i]
    elif select_node >=8 :
        for i in range(len(struc_B[2])):
            struc_B[2][i][select_node-7] = arr_A[i]
            struc_A[2][i][select_node-7] = arr_B[i]

    mutant = rd.random()
    #Mutant
    if(mutant > 0.001):
        if(select_node >= 2 and select_node < 5):
            i = int(rd.randrange(0,29))
            num = int(rd.randrange(0,1))
            if(num == 0):
                struc_B[0][i][select_node-2] = rd.random()
            else:
                struc_A[0][i][select_node-2] = rd.random()
        elif (select_node >= 5 and select_node < 8 ):
            i = int(rd.randrange(0,2))
            num = int(rd.randrange(0,1))
            if(num == 0):
                struc_B[1][0][i][select_node-5] = rd.random()
            else:
                struc_A[1][0][i][select_node-5] = rd.random()
        elif select_node >=8 :
            i = int(rd.randrange(0,1))
            num = int(rd.randrange(0,1))
            if(num == 0):
                struc_A[2][i][select_node-7] = rd.random()
            else:
                struc_B[2][i][select_node-7] = rd.random()
        
    return struc_A,struc_B 

def Selector(A = [] ,B = []):
    index_A = rd.randrange(0,len(A))
    index_B = rd.randrange(0,len(B)) 
    return index_A,index_B

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
    return sum_error/2.0


def map_to_NN(input_= [] , design = 0 , str_hid =[],int_node = [],hid_node = [],out_node = [],layer_1 = [],layer_hid= [],layer_end= []):
    output_data = [design,design]
    # print("struc :", str_hid)
    # print("hid :",hid_node)
    # print("input :",input_)
    input_ = Normalizing_of_Input(input_)
    # print("input :",input_)
    int_node,hid_node,out_node,layer_1,layer_hid,layer_end = forwardComputation(input_,str_hid,out_node,int_node,hid_node,out_node,layer_1,layer_hid,layer_end)
    # print("output :",out_node)
    sum_e = error_of_result(output_data,out_node)
    return sum_e

def Evaluate_Fitness(list_individual = [], design = 0,struc = [],int_node = [],hid_node = [],out_node = [],layer_1 = [],layer_hid= [],layer_end= []):
    fitness_arr = []
    list_struc = []
    for i in range(len(list_individual)):
        struc_ = []
        x = list_individual[i].split(",")
        _n = x[31].split("\n")
        x[31] = _n[0]
        select_w = x[2:]
        layer_1,layer_hid,layer_end,int_node,hid_node,out_node,bias = make_structure_NN(30,struc,2)
        fitness_arr.append(map_to_NN(select_w,design,struc,int_node,hid_node,out_node,layer_1,layer_hid,layer_end))
        struc_.append(layer_1)
        struc_.append(layer_hid)
        struc_.append(layer_end)
        struc_.append(int_node)
        struc_.append(hid_node)
        struc_.append(out_node)
        list_struc.append(struc_)
    return list_struc,fitness_arr
def Evaluate_Fitness_valid(list_individual = [], design = 0,struc = [],int_node = [],hid_node = [],out_node = [],layer_1 = [],layer_hid= [],layer_end= [],struc_type = []):
    fitness_arr = []
    for i in range(len(list_individual)):
        int_node = struc_type[i][3]
        hid_node = struc_type[i][4]
        out_node = struc_type[i][5]
        layer_1 = struc_type[i][0]
        layer_hid = struc_type[i][1]
        layer_end = struc_type[i][2]
        x = list_individual[i].split(",")
        _n = x[31].split("\n")
        x[31] = _n[0]
        select_w = x[2:]
        fitness_arr.append(map_to_NN(select_w,design,struc,int_node,hid_node,out_node,layer_1,layer_hid,layer_end))
    return fitness_arr

def Apply_Selection():
    return 0 
def Terminate():
    return 0 
def classifi(listarr):
    a = []
    b = []

    for i in range(len(listarr)):
        x = listarr[i].split(',')
        # print(x[1])
        if(x[1] == 'M'):
            a.append(listarr[i])
        elif(x[1] == 'B'):
            b.append(listarr[i])
    
    return a,b
def Read_Data(directory = ''):
    filepath = directory
    List_data = []
    with open(filepath) as fp:
        line = fp.readline()
        List_data.append(line)
        cnt = 1
        while line:
            List_data.append(line)
            line = fp.readline()
            cnt += 1
    return List_data

def main():
    Fitness_A = []
    Fitness_B = []
    structure_A = []
    structure_B = []
    list_data = Read_Data('C:/Users/Ninja/Work/4 year 1 term/Computer Intelligence/neural_networks/assignment3/data.txt')
    train , test  = select_input(len(list_data),90,10)
    # print("train :" , train[1])
    # print("test :" , test[1])
    data_train = []
    data_test = []
    for i in range(len(train[1])):
        data_train.append(list_data[train[1][i]])
    for i in range(len(test[1])):
        data_test.append(list_data[test[1][i]])
    # initial_pop
    A_Type,B_Type = classifi(data_train)
    print("-------------- initial_pop -----------------")
    print("type_A :",len(A_Type))
    print("type_B :",len(B_Type))
    
    # fitness_forIndiidual
    print("-------------- fitness -----------------")   
    output = 2
    # show_output_num = 0
    str_hid = [2,3,3] #hidden_structure [จำนวนlayer,node1,node2,...]
    Alayer_1,Alayer_hid,Alayer_end,Aint_node,Ahid_node,Aout_node,Abias = make_structure_NN(30,str_hid,output)
    structure_A,Fitness_A = Evaluate_Fitness(A_Type,0.5,str_hid,Aint_node,Ahid_node,Aout_node,Alayer_1,Alayer_hid,Alayer_end)
    Blayer_1,Blayer_hid,Blayer_end,Bint_node,Bhid_node,Bout_node,Bbias = make_structure_NN(30,str_hid,output)
    structure_B,Fitness_B = Evaluate_Fitness(B_Type,0.5,str_hid,Bint_node,Bhid_node,Bout_node,Blayer_1,Blayer_hid,Blayer_end)

    df1 = pd.DataFrame(np.array(Fitness_A))
    df2 = pd.DataFrame(np.array(Fitness_B))
    df3 = pd.DataFrame(np.array(structure_A))
    df4 = pd.DataFrame(np.array(structure_B))
    # Fitness_B = Evaluate_Fitness(B_Type,0.9,str_hid)
    print("======A======")
    print(df1)
    print("======B======")
    print(df2)
    print("======C======")
    print(df3)
    print("======D======")
    print(df4)
    
    Genaration = 0
    print("====== GA ======")
    while True :
        print("genaration :",Genaration)
        indA,indB = Selector(Fitness_A,Fitness_B)
        offspring_A,offspring_B = Create_Offsring(Fitness_A[indA],structure_A[indA],Fitness_B[indB],structure_B[indB])
        structure_A[indA] = offspring_A
        structure_B[indB] = offspring_B
        Genaration += 1
        print("----------------------------------")
        if (Check_Stop(Genaration) == False ):
            break

    # ---------------- valid ------------
    A_Type_test,B_Type_test = classifi(data_test)
    print("-------------- initial_pop -----------------")
    print("type_A :",len(A_Type_test))
    print("type_B :",len(B_Type_test))
    # fitness_forIndiidual
    print("-------------- fitness -----------------")  
    Fitness_A = Evaluate_Fitness_valid(A_Type_test,0.1,str_hid,Aint_node,Ahid_node,Aout_node,Alayer_1,Alayer_hid,Alayer_end,structure_A) 
    Fitness_B = Evaluate_Fitness_valid(B_Type_test,0.9,str_hid,Bint_node,Bhid_node,Bout_node,Blayer_1,Blayer_hid,Blayer_end,structure_B)
    df1 = pd.DataFrame(np.array(Fitness_A))
    df2 = pd.DataFrame(np.array(Fitness_B))
    num = 0
    for i in range(len(Fitness_B)):
        num += Fitness_B[i]
    num = num/len(Fitness_B)
    
    # Fitness_B = Evaluate_Fitness(B_Type,0.9,str_hid)
    print("======A======")
    print(df1)
    print("======B======")
    print(df2)
    print("MSE AVG :",num)
  
    return 0 

def Check_Stop(GB = 0 ):
    boo = True
    if(GB >= Genararion_number):
        boo = False

    return boo
if __name__ == "__main__":    
    main()