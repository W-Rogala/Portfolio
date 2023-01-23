'''Napisać program równoległy, któnorma ry wyznaczy normę wektoraA[i] (i=1,1024). 
Tablicę należy wypełnic liczbami losowymi z przedziału [0:1]. 
Wynik należy zabrać na procesorze 0 i wypisać na standardowe wyjście. 
Dla przypomnienia, norma wektorów  to √∑A[i]*A[i].
'''
from multiprocessing import Process, Array
import numpy as np
import time
import matplotlib.pyplot as plt

n = 1024 

# fuction computing squares of values, meant to be parallelized
# 'A' is a input and 'result' is an output parameter which is shared between processes
# and stores final result

def squares(A, result):
	for idx,i in enumerate(A):
		result[idx] = i*i


if __name__  == '__main__':
	processes = []
	
	A = np.random.rand(n) #vector consisting of random values between 0 and 1
	result = Array('d',n) # array as a memory shared between processes,
	# where 'd' denotes double, and 'n' - dimension of given array 

	start_time = time.time()

	process = Process(target=squares, args=(A,result))
	process.start()
	process.join()

	stop_time = time.time()
	print("Calculation time within parallel processing",stop_time - start_time,"seconds.")

	# norm of vector calculated basing on elements calculated parallely
	# stored in shared memory

	Norm_of_vec = np.sqrt(sum(result)) 
	stop_time = time.time()


	print("Total calculation time is",stop_time - start_time,"seconds.")
	print("Norm of vector is equal=",Norm_of_vec)