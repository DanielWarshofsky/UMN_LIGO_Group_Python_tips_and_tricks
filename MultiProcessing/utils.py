from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import numpy as np
from astropy.coordinates import SkyCoord
import time


def summary(name,timeit_results):
    r=np.array(timeit_results)
    print(f'{name} {r.size} runs: \n Mean {r.mean()} \n Standard deviation {r.std()}')

def faster_integral(a): #some numpy magic
    resolution=1000 #number of rectangles
    grid=np.linspace(0,1,resolution)
    result=((1+grid)**a)* (1/resolution)
    return result.sum()

def simple_loop_index(vals): #do a simple loop and index result to the list
    results=np.ones(len(vals))
    for i,val in enumerate(vals):
        results[i]=faster_integral(val)
    return results
    
def built_in_process(vals,n_threads=4):
    tp=ProcessPoolExecutor(n_threads)
    #now batch up vals
    batch_size=vals.shape[0]//n_threads
    futures=[]
    for i in range(n_threads-1): # the first few with proper batch size
        future=tp.submit(simple_loop_index,vals[i*batch_size:(i+1)*batch_size])
        futures.append(future)
    #now the last one witht the remainder
    futures.append(tp.submit(simple_loop_index,vals[(n_threads-1)*batch_size:-1]))
    #finally collect the data and combine into one list
    results_list=[f.result() for f in futures] #get the result of the threads
    #this function outputs a numpy array so we can just
    return np.concatenate(results_list)

def built_in_threads(vals,n_threads=4):
    tp=ThreadPoolExecutor(n_threads)
    #now batch up vals
    batch_size=vals.shape[0]//n_threads
    futures=[]
    for i in range(n_threads-1): # the first few with proper batch size
        future=tp.submit(simple_loop_index,vals[i*batch_size:(i+1)*batch_size])
        futures.append(future)
    #now the last one witht the remainder
    futures.append(tp.submit(simple_loop_index,vals[(n_threads-1)*batch_size:-1]))
    #finally collect the data and combine into one list
    results_list=[f.result() for f in futures] #get the result of the threads
    #this function outputs a numpy array so we can just
    return np.concatenate(results_list)

def future_dec(func,n_jobs=4):
    #func must have first arg as a list or list like.
    def inner(*args,**kwargs):
        data=args[0]
        tp=ProcessPoolExecutor(n_jobs)
        #now batch up vals
        batch_size=len(data)//n_jobs

        batches=[data[i*batch_size:(i+1)*batch_size] for i in range(n_jobs)]
        # add on the remainder
        if len(data)%n_jobs!=0: batches.append(data[n_jobs*batch_size:len(data)])

        futures=[]
        for batch in batches:
            future=tp.submit(func,batches,*args[1:-1],**kwargs)
            futures.append(future)

        results_list=[f.result() for f in futures] #get the result of the threads
        return results_list
    return inner

def test_func(a):
    time.sleep(.2)
    return a

if __name__=='__main__':
    #how to use future_dec
    
    #step 1 have you NON-LOCALLY defined function, here test_func()
    test_func(1)
    # make sure that the first arg of your function is the one you want to batch over
    
    #step 2 create a new function with the decorator
    mp_test_func=future_dec(test_func,n_jobs=4)
    #step 3 use the new function
    b=np.ones(10)*0.1
    mp_test_func(b)
