import numpy as np
import copy
import random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd



###################### GA ######################  
class SWARM:
    def __init__(self, data_trainingset, desire_traingset):
        ########## data ###########
        
        np.random.seed(1)
        self.data = data_trainingset
        self.desire = desire_traingset
        self.input = np.asarray(copy.deepcopy(data_trainingset))
        
        ############### create Weight #############
        self.weight = []
        self.bias = []
        self.deltaweight = []
        self.deltabias = []
        self.weight_old1 = []
        self.weight_old2 = []

        ################ create layer ###############
        self.countLayer = 0
        self.Node = []

        ################### feed forward ############
        self.output_feedforward = []

        ################# fitness list ################
        self.npoppu = 0 
        self.fitnessplot = []
        self.chromosome = {}
        self.child = []
        self.Loss = []
        self.pop_list = []
        self.velocity = []
        self.Pbest = []
        self.Gbest = [[10000,0]]

    ############# NN ####################
    def create_weight(self):
        for i in range(len(self.Node)):
            if ( i == 0 ):
                self.weight.append(2*np.random.rand(len(self.input), self.Node[i]) - 1)
                self.bias.append(2*np.random.rand(self.Node[i]) - 1)
                self.deltaweight.append(np.ones((len(self.input), self.Node[i])))
                self.deltabias.append(np.ones(self.Node[i]))
                self.weight_old1.append(np.zeros((len(self.input), self.Node[i])))
                self.weight_old2.append(np.zeros((len(self.input), self.Node[i])))
            else:
                self.weight.append(2*np.random.rand(self.Node[i-1], self.Node[i]) - 1)
                self.bias.append(2*np.random.rand(self.Node[i]) - 1)
                self.deltaweight.append(np.ones((self.Node[i-1], self.Node[i])))
                self.deltabias.append(np.ones(self.Node[i]))
                self.weight_old1.append(np.zeros((len(self.input), self.Node[i])))
                self.weight_old2.append(np.zeros((len(self.input), self.Node[i])))

    def Layer(self, node):
        self.countLayer+=1
        self.Node.append(node)
    
    ################# feedforward -> error ################
    def sigmoid(self, v):
        return 1/(1+np.exp(-v))
    def FeedForward(self, chromosome): 
        self.output_feedforward = []
        self.output_feedforward.append(self.input.T)
        out = np.array(self.output_feedforward[0])
        for i in range(len(self.Node)): # feed in each layer
            v = np.dot(copy.deepcopy(chromosome[i].T), copy.deepcopy(self.output_feedforward[i]))
            out = self.sigmoid(v)
            self.output_feedforward.append(out)
        
        return out

    ################ find fitness ################
    def create_pop(self): # return list
        weight = []
        for  i in range(len(self.Node)):
            if (i == 0):
                weight.append(np.random.uniform(-1, 1, (len(self.data[0]), self.Node[i])))
            else : 
                weight.append(np.random.uniform(-1, 1, (self.Node[i-1], self.Node[i])))
            
        return weight
    def find_fitness(self, mae): 
        return np.mean(np.abs(mae))
    def fitness_list(self, npoppu, generation):
        #print("ok")
        self.npoppu = npoppu
        self.fitnessplot = []

        for k in range(generation) :
            fitne = []
            # self.chromosome = {}
            for i in range(npoppu): 
                if k == 0 :
                    pop = self.create_pop()
                    pop = np.asarray(pop)
                    
                    out = self.FeedForward(pop)
                else :
                    
                    out = self.FeedForward(copy.deepcopy(self.pop_list[i][1]))

                self.Loss = []

                for j in range(len(self.data)): 
                    # print(out[0][j])
                    # print(copy.deepcopy(self.desire[j]))
                    err = out[0][j] - copy.deepcopy(self.desire[j])
                    self.Loss.append(err)

                # mse = np.asarray(copy.deepcopy(self.Loss))
                # mse = (mse*mse)/ 2
                # mse = np.mean(mse) # scalar
                mae = np.asarray(copy.deepcopy(self.Loss)) ######-> sum |e|/n
                fitness = self.find_fitness(mae)
                fitne.append(fitness)
                #print("fitness",fitne)

                ################### first #################
                if k == 0 :
                    self.pop_list.append([fitness,pop])
                    self.velocity.append(copy.deepcopy(self.create_pop()))
                    #print(copy.deepcopy(self.pop_list[i]))
                    self.Pbest.append(copy.deepcopy(self.pop_list[i]))
                    
                ################### Pbest ##############
                if fitness <self.Pbest[i][0]:
                    self.Pbest[i][0] = copy.deepcopy(fitness)
                    self.Pbest[i][1] = copy.deepcopy(self.pop_list[i][1])
                #################### Gbest ################
                if fitness < self.Gbest[0][0]:
                    self.Gbest[0][0] = copy.deepcopy(fitness)
                    self.Gbest[0][1] = copy.deepcopy(self.pop_list[i][1])
                ################### velocity ################
                
                x_pbest = copy.deepcopy(self.Pbest[i][1])
                x_t = copy.deepcopy(self.pop_list[i][1])
                x_gbest = copy.deepcopy(self.Gbest[0][1])
                v_t1 = self.velocity[i]
                ################ random p1,p2############
                if  k  + i ==0 :
                    r1 = random.randint(0,1)
                    c1 = random.randint(0,2)
                    p1 = r1*c1

                    r2 = random.randint(0,1)
                    c2 = random.randint(0,2)
                    p2 = r2*c2

                self.velocity[i] = copy.deepcopy(self.velocity[i])+p1*(x_pbest-x_t)+p2*(x_gbest-x_t)
                v_t = copy.deepcopy(self.velocity[i])
                ################ poistion ################
                # print("x_t",x_t)
                # print("v_t",v_t)
                self.pop_list[i][1] =  x_t +v_t
            fitne = np.asarray(copy.deepcopy(fitne))
            fitne = np.mean(fitne)
            #print(fitne)``
            #print("Fitness {}" .format(k+1, fitne))
            self.fitnessplot.append(fitne)
            # self.chromosome = self.matingPool()
            # n_crossingpoint = np.random.randint(0, npoppu-1)
            # child = self.Crossover(n_crossingpoint)
            # self.child = self.mutate(child)


    

    

    def Predict (self, data_test, desire_test):
        data_test = np.asarray(copy.deepcopy(data_test))
        desire_test = np.asarray(copy.deepcopy(desire_test))

        self.input = data_test
        fitness_value = []
        for i in range(len(self.pop_list)):

            output = self.FeedForward(self.pop_list[i][1])

            Loss = []
            for j in range(len(desire_test)):
                err = output[0][j] - desire_test[j]
                Loss.append(err)


            mae = np.asarray(copy.deepcopy(Loss))
            # mae = (mse*mse)/ 2
            # mse = np.mean(mse)
            fitness = self.find_fitness(mae)
            fitness_value.append(fitness)

        fitness_value = np.asarray(copy.deepcopy(fitness_value))
        fitness_value = (np.mean(fitness_value))
        print("Fitness_value = {}" .format(fitness_value))                

    def plotFitness(self, fold):
        fitness = self.fitnessplot
        fig, ax = plt.subplots()
        ax.plot(range(1, len(fitness)+1), fitness)
        ax.set(xlabel='iteration', ylabel='Fitness', title='Fold {}' .format(fold))
        fig.savefig("Fold {}.png" .format(fold))




######## Main ################
def main():
    ######### get data ############
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
    #print(data)

    ################## cross validation ###################
    fold = 10
    data_trainingset = []
    desire_trainingset=[]
    data_testset=[]
    desire_testset=[]

    
    crossvali = int(len(data)*fold/100)

    for i in range(fold):
        if i == (fold-1) :
            data_testset.append(data[0+i*crossvali:len(data)])
            desire_testset.append(desire[0+i*crossvali:len(desire)])

            data_trainingset.append(data[0:0+i*crossvali])
            desire_trainingset.append(desire[0:0+i*crossvali])
        else:
            
            
            x_trainingset1 = data[0:i*crossvali]
            y_trainingset1 = desire[0:i*crossvali]
            
            
            x_trainingset2 = data[crossvali*(i+1):len(data)]
            y_trainingset2 = desire[crossvali*(i+1):len(desire)]

            data_trainingset.append(x_trainingset1 + x_trainingset2)
            desire_trainingset.append(y_trainingset1 + y_trainingset2)


            data_testset.append(data[0+i*crossvali:crossvali+i*crossvali])
            desire_testset.append(desire[0+i*crossvali:crossvali+i*crossvali])
        
    for i in range(len(data_trainingset)): #### Fold
            print("Fold {}" .format(i+1))
            # print(len(data_trainingset[i]))
            # print(desire_trainingset[i])
            neural = SWARM(data_trainingset[i], desire_trainingset[i])
            neural.Layer(70)
            neural.Layer(1)
        
            neural.fitness_list(50, 100)
            neural.plotFitness(i+1)
            neural.Predict(data_testset[i],desire_testset[i])

            "-----------------------------"


            



if __name__ == "__main__":
    main()


