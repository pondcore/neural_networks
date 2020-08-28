import numpy as np
import random
from numpy import genfromtxt
my_data = np.genfromtxt('data.csv', dtype=int, delimiter='\t', names=True)
data = [i for i in np.array(my_data)]

if __name__ == "__main__":

    for i in range(len(data)):
        for j in range(8):
            print(data[i][j],end='   ')
        print('\n')
    pass