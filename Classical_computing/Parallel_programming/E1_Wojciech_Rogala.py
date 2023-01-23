'''
Napisać równoległa implementację operacji dodawania wektora B i wektora C 
Wynik zapisywany jest w wektorze A. Tzn: A[i] = B[i] * C[i], i=0....n. n=1048576.
Waktory B i C należy wypełnić losowymi liczbami z zakresu [0:1]. 
Jako wynik wypisać A[0] i A[n-1] - należy zwrócić uwagę gdzie składowane są odpowiednie elementy. Wektory mogą być w dowolny sposób rozmieszczone w pamięci. 

Aplikacje należy uruchomić na 1, 2, 4 procesorach (rdzeniach, wątkach) (może być na laptopie)
 i podać czasy wykonania.
'''


from multiprocessing.dummy import Pool as ThreadPool 
from time import time
import numpy as np

n = 1048576
num_threads = [1,2,4] 


def sum_of_vectors(A,B):
	return A+B

if __name__ == '__main__':
	#gneration of two vencors with random values between 0 and 1 
	A = np.random.rand(n)
	B = np.random.rand(n)

	#for loop over number of threads to test performance 
	# on 1, 2 4 threads 
	for threads in num_threads:
		pool = ThreadPool(threads)
		start_time = time()

		# starmap method maps multiple variable dependent functions
		sum_pool_result  = pool.starmap(sum_of_vectors, zip(A,B)) 
		pool.close() 
		pool.join()


		end_time=time()

		print("Calculation using number of cores equal ", threads, ", took", end_time-start_time, "seconds.")
