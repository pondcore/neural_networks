import matplotlib.pyplot as plt
import numpy as np
import random as rd
import pandas as pd
import math

# defind variable
Genararion_number = 10000
v_struc =[] 
w_t_1_ = []
layer_1 = []
output_data =[]
out_node = []
hid_node = []
layer_hid = []
layer_end = []
position = [] 
velocity = [] 
x_pbest = [0,0]
x_gbest = [0,0]

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
    if(mutant > 0.8):
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
    # input_ = Normalizing_of_Input(input_)
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
    fitness_arr = 0
    fitness_arr= map_to_NN(list_individual,design,struc,int_node,hid_node,out_node,layer_1,layer_hid,layer_end)
    return fitness_arr

def Apply_Selection():
    return 0 
def Terminate():
    return 0 
def find_fitness(mae): 
        return np.mean(np.abs(mae))
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
def find_Pbest(pbest = [], pos = [] , p = 0, v_ = [],gbest = 0 ):
    pbest_ = []
    x_pbest = [0,0]
    x_gbest = [0,0]
    # global x_gbest
    r1 = rd.randint(0,1)
    c1 = rd.randint(0,2)
    p1 = r1*c1

    r2 = rd.randint(0,1)
    c2 = rd.randint(0,2)
    p2 = r2*c2
    for i in range(len(pbest)):
        if(p == 0):
            pbest[i] = rd.random()
            # pos.append(pos[i])
            # v_.append(rd.random())
        else:
            # print(find_fitness(pos[i]),"===",pbest[i])
            if find_fitness(pos[i]) < pbest[i]:
                pbest[i] = find_fitness(pos[i])
                x_pbest = pos[i]
                # print("yaaaaa!!!")
            if find_fitness(pos[i]) < gbest:
                gbest = find_fitness(pos[i])
                x_gbest = pos[i]
            # print(v_[i])
            v_[i][0] = v_[i][0]+p1*(x_pbest[0]-pos[i][0])+p2*(x_gbest[0]-pos[i][0])
            v_[i][1] = v_[i][1]+p1*(x_pbest[1]-pos[i][1])+p2*(x_gbest[1]-pos[i][1])

            



    return  pbest,pos,v_
def find_Gbest(list_ = [], pos = [] , p = 0):
    pbest_ = []
    if(p == 0):
        for i in range(len(list_)):
           pbest_.append(pos[i])

    else:
        pbest_ = []
    return  pbest_
def main():
    Fitness_A = []
    Fitness_B = []
    structure_A = []
    structure_B = []
    Pbest = []
    Gbest = []
    data = 'C:/Users/Ninja/Work/4 year 1 term/Computer Intelligence/neural_networks/assignment4/AirQualityUCI.xlsx'
    list_data = []
    
    list_data = pd.read_excel(data) 
    #print(list_data)
    ############### normalize################
    count = 0
    data = []
    desire = []

    for i in list_data.columns :
        
        if  count == 3 or count == 6 or count == 8 or count >= 10 :
            nor = list_data[i]
            ########### normalization#############
            min_data = min(nor)
            max_data = max(nor)
            norma=[]
            for j in range (len(nor)):
                normalize = 0
                normalize = (nor[j]-min_data)/(max_data-min_data)
                norma.append(normalize)
            data.append(norma)
            #print(data)
        elif count == 5 :
            nor = list_data[i]
            ########### normalization#############
            min_data = min(nor)
            max_data = max(nor)
            norma=[]
            for j in range (len(nor)):
                normalize = 0
                normalize = (nor[j]-min_data)/(max_data-min_data)
                norma.append(normalize)
            #print(norma)
            desire = norma
            #print(desire)
        count += 1 
    #print(data[0])
    #print("desire_1",desire)
    data = list(map(list, zip(*data)))
    df1 = pd.DataFrame(np.array(data))

    print(df1)
    train , test  = select_input(len(data),90,10)
    data_train = []
    data_test = []
    for i in range(len(train[1])):
        data_train.append(data[train[1][i]])
    for i in range(len(test[1])):
        data_test.append(data[test[1][i]])
    print("-------------- data_train -----------------")
    df1 = pd.DataFrame(np.array(data_train))
    print(df1)
    print("-------------- data_test -----------------")
    df1 = pd.DataFrame(np.array(data_test))
    print(df1)
    output = 2
    str_hid = [2,3,3] #hidden_structure [จำนวนlayer,node1,node2,...]
    Alayer_1,Alayer_hid,Alayer_end,Aint_node,Ahid_node,Aout_node,Abias = make_structure_NN(7,str_hid,output)
    for  j in range(10):
        Gbest = 0
        print("P(",j+1,")")
        if (j == 0):
            for i in range(len(data_train)):
                # print("X :", i ," :" )
                # fitness_forIndiidual
                
                Fitness_A.append(Evaluate_Fitness_valid(data_train[i][:7],data_train[i][7]
                ,str_hid,Aint_node,Ahid_node,Aout_node,Alayer_1,Alayer_hid,Alayer_end))
                Pbest.append(0)
                po = [rd.uniform(-1,1),rd.uniform(-1,1)]
                global position
                position.append(po)
                vec = [rd.random(),rd.random()]
                global velocity
                velocity.append(vec)

            # print("-------------- Position -----------------")
            # df1 = pd.DataFrame(np.array(position))
            # print(df1)
            Pbest,position,velocity = find_Pbest(Fitness_A,position,j,velocity)  
            
            # print("-------------- Pbest -----------------")
            # df1 = pd.DataFrame(np.array(Pbest))
            # print(df1)

            # # Gbest = find_Gbest(Fitness_A,position,j)
            # print("-------------- Gbest -----------------")   
            # df1 = pd.DataFrame(np.array(position))
            # print(df1)
        else:
            # for i in range(10):
            #     print("X :", i ," :" )
            #     # fitness_forIndiidual
                
            #     Fitness_A[i] = Evaluate_Fitness_valid(data_train[i][:7],data_train[i][7],str_hid,Aint_node,Ahid_node,Aout_node,Alayer_1,Alayer_hid,Alayer_end)
            #     # Pbest.append(0)
            # print("-------------- 1 -----------------")
            # df1 = pd.DataFrame(np.array(Pbest))
            # print(df1)
            # df1 = pd.DataFrame(np.array(position))
            # print(df1)
            # Fitness_A.append(Evaluate_Fitness_valid(data_train[i][:7],data_train[i][7],str_hid,Aint_node,Ahid_node,Aout_node,Alayer_1,Alayer_hid,Alayer_end))
            Pbest,position,velocity = find_Pbest(Pbest,position,j,velocity) 
        
            position = position + velocity
            
    # for i in range(len(data_test)):
    #     Pbest,position,velocity = find_Pbest(Pbest,position,j,velocity) 
    # Pbest[0:len(data_test)]
    print("-------------- velocity -----------------")
    df1 = pd.DataFrame(np.array(velocity[0:935]))
    print(df1)
    print("-------------- Position_test -----------------")
    df1 = pd.DataFrame(np.array(position[0:935]))
    print(df1)

    return 0 

def Check_Stop(GB = 0 ):
    boo = True
    if(GB >= Genararion_number):
        boo = False

    return boo
if __name__ == "__main__":    
    main()