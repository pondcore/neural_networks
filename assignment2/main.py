import matplotlib.pyplot as plt
import numpy as np
import random as rd
plt.rcParams['font.family'] = 'Tahoma'

temperature_degree = []
moisture_degree = []
membershipe_T = 0
membershipe_M = 0

#range-plotgraph 
ylist = range(0,1)
mlist = range(0,100)
tlist = range(10,50)

def Input_random(str_):
    rand = 0
    if(str_ == 'T'):
        rand = rd.randrange(10,50)
    elif(str_ == 'M'):
        rand = rd.randrange(0,100)
    return rand

def moisture(str_,value_, num_):
    membershipe_M = 0
    if(str_ == 'down'):
        if(num_ <= value_):
            membershipe_M = 1
        elif (num_ > value_ and num_ < value_ + 25):
            membershipe_M = num_ *(-1/25) + (1 + value_/25)
        else:
            membershipe_M = 0
    elif (str_ == 'mid'):
        if(num_ <= value_ - 25):
            membershipe_M = 0
        elif (num_ > value_ - 25 and num_ < value_):
            membershipe_M = num_ *(1/25) + (1 - (value_ /25))
        elif (num_ >= value_ and num_ < value_ + 25):
            membershipe_M = num_ *(-1/25) + (1 + (value_/25))
        else:
            membershipe_M = 0
    elif (str_ == 'up'):
        if(num_ <= value_ - 25):
            membershipe_M = 0
        elif (num_ >  value_ - 25 and num_ < value_  ):
            membershipe_M = num_*(1/25) + (1 - (value_/25))
        else:
            membershipe_M = 1
    return membershipe_M

def temperature(str_,value_, num_):
    membershipe_T = 0
    if(str_ == 'down'):
        if(num_ <  value_ ):
            membershipe_T = 1
        elif (num_ >= value_ and num_ < value_ + 5):
            membershipe_T = (value_ + 5 - num_)*(1/5)
        else:
            membershipe_T = 0
    elif (str_ == 'mid'):
        if(num_ < value_ - 5):
            membershipe_T = 0
        elif (num_ > value_ - 5 and num_ < value_):
            membershipe_T =  num_ *(1/5) + (1 - (value_ /5))
        elif (num_ >= value_ and num_ < (value_ + 5)):
            membershipe_T = num_ *(-1/5) + (1 + (value_/5)) 
        else:
            membershipe_T = 0
    elif (str_ == 'up'):
        if(num_ <= value_ - 5):
            membershipe_T = 0
        elif (num_ >  value_ - 5 and num_ < value_  ):
            membershipe_T = num_*(1/5) + (1 - (value_/5))
        else:
            membershipe_T = 1
    return membershipe_T

def mandani(input_):
    Min_Alpha = [0,0,0]
    V_ = [0]
    F_ = [0,0,0,0,0,0]
    N_ = [0,0]
    # Veryfresh_num = rd.random(0,5)
    # Fresh_num = rd.random(0,5)
    # Notfresh_num = rd.random(0,5)
    # rule
    V_[0] = min(temperature('down',25,input_[0]),moisture('up',75,input_[1]))
    F_[0] = min(temperature('down',25,input_[0]),moisture('down',25,input_[1]))
    F_[1] = min(temperature('down',25,input_[0]),moisture('mid',50,input_[1]))
    F_[2] = min(temperature('mid',30,input_[0]),moisture('down',25,input_[1]))
    F_[3] = min(temperature('mid',30,input_[0]),moisture('mid',50,input_[1]))
    F_[4] = min(temperature('mid',30,input_[0]),moisture('up',75,input_[1]))
    F_[5] = min(temperature('up',35,input_[0]),moisture('down',25,input_[1]))
    N_[0] = min(temperature('up',35,input_[0]),moisture('mid',50,input_[1]))
    N_[1] = min(temperature('up',35,input_[0]),moisture('up',75,input_[1]))

    Min_Alpha[0] = V_
    Min_Alpha[1] = F_
    Min_Alpha[2] = N_
    D_ = 0.0
    Max_ = []
    for i in range(len(Min_Alpha)):
        Max_.append(max(Min_Alpha[i]))
    #defuzzi
    dez = np.zeros(2)

    for i in range(len(Min_Alpha)):
        for j in range(len(Min_Alpha[i])):
            dez[0] += Min_Alpha[i][j]*(i+1)
            dez[1] += Min_Alpha[i][j]
    D_ = dez[0]/dez[1]
    if ( D_ >= 1 and D_ < 2):
        if D_ - 1 != 0 :
            print("High: "+ str((D_ - 1)*100.0 ) +" %" )
        else :
            print("High")
    elif ( D_ >= 2 and D_ < 3 ) :
        if D_ - 2 != 0 :
            print("Normal:"+ str((D_ - 2)*100.0 ) +" %" )
        else :
            print("Normal")
    elif ( D_ >= 3 ) :
        if D_ - 3 != 0 :
            print("Low:"+ str((D_ - 3)*100.0)+" %" )
        else :
            print("Low")

    return 0


def main():
    random_input = [0,0]
    Num_input = 20
    for i in range(Num_input):
        random_input[0] = Input_random('T')
        random_input[1] = Input_random('M')
        print(random_input)
        mandani(random_input)
        print('----------------------')



    graph()


    return 0

def fresh(str_,value_, num_):
    membershipe_M = 0
    if(str_ == 'down'):
        if(num_ <= value_):
            membershipe_M = 1
        elif (num_ > value_ and num_ < value_ + 25):
            membershipe_M = num_ *(-1/25) + (1 + value_/25)
        else:
            membershipe_M = 0
    elif (str_ == 'mid'):
        if(num_ <= value_ - 25):
            membershipe_M = 0
        elif (num_ > value_ - 25 and num_ < value_):
            membershipe_M = num_ *(1/25) + (1 - (value_ /25))
        elif (num_ >= value_ and num_ < value_ + 25):
            membershipe_M = num_ *(-1/25) + (1 + (value_/25))
        else:
            membershipe_M = 0
    elif (str_ == 'up'):
        if(num_ <= value_ - 25):
            membershipe_M = 0
        elif (num_ >  value_ - 25 and num_ < value_  ):
            membershipe_M = num_*(1/25) + (1 - (value_/25))
        else:
            membershipe_M = 1
    return membershipe_M

def graph():

    fig, axs = plt.subplots(3,1)

    t1 = [temperature('down',25,x) for x in tlist]
    t2 = [temperature('mid',30,x) for x in tlist]
    t3 = [temperature('up',35,x) for x in tlist]

    axs[0].set_title(label=u"อุณหภูมิ")
    axs[0].plot(tlist, t1, label=u'อุณหภูมิต่ำ')
    axs[0].plot(tlist, t2, label=u'อุณหภูมิปานกลาง')
    axs[0].plot(tlist, t3, label=u'อุณหภูมิสูง')
    axs[0].set_xlabel('T')
    axs[0].set_ylabel('Membership')
    axs[0].legend()

    m1 = [moisture('down',25,x) for x in mlist]
    m2 = [moisture('mid',50,x) for x in mlist]
    m3 = [moisture('up',75,x) for x in mlist]
    
    axs[1].set_title(label=u"ความชื้น")
    axs[1].plot(mlist, m1, label=u'ความชื้นต่ำ')
    axs[1].plot(mlist, m2, label=u'ความชื้นปานกลาง')
    axs[1].plot(mlist, m3, label=u'ความชื้นสูง')
    axs[1].set_xlabel('%')
    axs[1].set_ylabel('Membership')
    axs[1].legend()

    m1 = [fresh('down',50,x) for x in mlist]
    m2 = [fresh('mid',70,x) for x in mlist]
    m3 = [fresh('up',90,x) for x in mlist]
    
    axs[2].set_title(label=u"ปกติ")
    axs[2].plot(mlist, m1, label=u'สูงมาก')
    axs[2].plot(mlist, m2, label=u'ปกติ')
    axs[2].plot(mlist, m3, label=u'ไม่สูง')
    axs[2].set_xlabel('%')
    axs[2].set_ylabel('Membership')
    axs[2].legend()

    plt.show()

    return 0 
if __name__ == "__main__":    
    main()