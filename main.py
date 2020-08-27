import numpy as np
from numpy import genfromtxt

if __name__ == "__main__":
    data = np.genfromtxt('data.csv', dtype=None, delimiter='\t', names=True)
    print(data)
    pass