'''
Napisać program równoległy, który wczyta z pliku in.txt n liczb całkowitych z zakresu [0:100],
a nastepnie w sposób równoległy wyznaczy histogram, 
tzn. wyznaczy wektor B[i] (i=0,100) w którego kolejnych komórkach 
będzie znajdowała się ilość liczb o okreslonej wartości B[0] ilość 0, B[1] ilość 1, 
B[2] ilość 2 itd. Tablicę B[i] należy zabrać na procesorze 0 i wypisać na standardowe wyjście.

Plik wejściowy powinien zawierać co najmniej 1024 liczb. 
'''

import cProfile
import pstats



from multiprocessing import Pool
from collections import Counter
import glob
import time
import numpy as np


# Creation of histogram generating function using 'Counter' method from collections library


def create_histograms(filepath):

    histogram = Counter() 
    with open(filepath) as f:
        for line in f:
            histogram[line.strip()] += 1
    return histogram

def singlethread():
    start_time = time.time()

    datafile_path = glob.glob("G:\\Wojtek\\ASUS\\DOKUMENTY\\STUDIA\\Computational_Engineering\\Parallel Programming\\in.txt") 

    pool = Pool(processes=1)

    histograms = pool.map(create_histograms, datafile_path) # mapping function on 1 processes

    pool.close()
    pool.join()


    # sorting histogram into B vector with dimmension = 101 spanned by integers 0,1,...,100, 
    #with values corresponding to number of repetitions of given value
    i = 0
    B=[]
    while str(i) in histograms[0]:
    	B.append(histograms[0][str(i)])
    	print('Value ', i, ' occurs ',B[i], ' times.')
    	i+=1
    print(B)

    whole_time = time.time()-start_time
    print("Calculation time ", whole_time," seconds")

def multithread():
    start_time = time.time()

    datafile_path = glob.glob("G:\\Wojtek\\ASUS\\DOKUMENTY\\STUDIA\\Computational_Engineering\\Parallel Programming\\in.txt") 

    pool = Pool(processes=8)

    histograms = pool.map(create_histograms, datafile_path) # mapping function on 4 processes

    pool.close()
    pool.join()


    # sorting histogram into B vector with dimmension = 101 spanned by integers 0,1,...,100, 
    #with values corresponding to number of repetitions of given value
    i = 0
    B=[]
    while str(i) in histograms[0]:
    	B.append(histograms[0][str(i)])
    	print('Value ', i, ' occurs ',B[i], ' times.')
    	i+=1
    print(B)

    whole_time = time.time()-start_time
    print("Calculation time ", whole_time," seconds")

if __name__ == "__main__":
    # Initialize profile class and call regression() function
    profiler = cProfile.Profile()
    profiler.enable()
    
    #Generate file with 10000 random values within range (0,100)
    a  = open("G:\\Wojtek\\ASUS\\DOKUMENTY\\STUDIA\\Computational_Engineering\\Parallel Programming\\in.txt",'w+')

    for i in range(10000000):
	    a.write( str(np.random.randint(low=0, high=101))+'\r\n')
    a.close()


    singlethread()
    multithread()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')

    # Print the stats report
    stats.print_stats()   



