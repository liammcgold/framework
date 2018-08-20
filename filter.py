import numpy as np
import time
import os
import psutil
import log
from multiprocessing.dummy import Pool as ThreadPool


def filter(seg,thresh):

    new_seg=seg

    uniques=np.unique(seg)

    dictionary={}
    spacing_dictionary={}
    for num in uniques:

        if num==0:
            spacing_dictionary[num]=0
        else:
            spacing_dictionary[num]=n
            n=n+10



    for num in uniques:
        dictionary[num]=0


    time_s=time.time()
    for i in range(0,np.shape(seg)[0]):
        for j in range(0,np.shape(seg)[1]):
            for k in range(0,np.shape(seg)[2]):
                dictionary[seg[i,j,k]]+=1
    time_c=time.time()-time_s
    print("time ",time_c)

    time_s = time.time()
    for i in range(0,np.shape(seg)[0]):
        for j in range(0,np.shape(seg)[1]):
            for k in range(0,np.shape(seg)[2]):
                if(dictionary[seg[i,j,k]]<thresh):
                    seg[i,j,k]=0

    time_c = time.time() - time_s
    print("time ", time_c)


    new_seg=more_even_spacing(seg)

    return new_seg


def top_n(seg,n):

    time_b = time.time()
    mem_b = psutil.virtual_memory().used




    uniques = np.unique(seg)

    dictionary = {}




    for num in uniques:
        dictionary[num] = 0

    time_s = time.time()
    for i in range(0, np.shape(seg)[0]):
        for j in range(0, np.shape(seg)[1]):
            for k in range(0, np.shape(seg)[2]):
                dictionary[seg[i, j, k]] += 1
    time_c = time.time() - time_s
    print("time ", time_c)

   # np.swapaxes(dictionary,0,1)


    list=np.zeros(n)
    full_list=np.zeros(np.shape(uniques)[0])
    for i in range(0,np.shape(uniques)[0]):
        full_list[i]=int(dictionary[uniques[i]])


    full_list=np.sort(full_list)



    list=full_list[-n:]


    dict_new={}


    for n in range(0,len(list)):
        dict_new[list[n]]=n+1

    spacing_dictionary = {}

    a = 10
    for num in uniques:
        if (dictionary[num] == list).any():
            if num == 0:
                spacing_dictionary[num] = 0
            else:
                spacing_dictionary[num] = a
                a = a + 10


    workers=1
    args=split_into_n_threads(workers,seg,dictionary,spacing_dictionary,list)

    pool=ThreadPool(workers)

    results=pool.starmap(fix_seg,args)

    print(np.shape(results))

    new_seg=unify_threads(results)


    time_e = time.time()
    mem_a = psutil.virtual_memory().used
    log.log("filter", np.shape(seg), time_e - time_b, mem_a - mem_b)




    return new_seg[0:np.shape(seg)[0]-1]



def more_even_spacing(seg):

    uniques=np.uniques(seg)

    dictionary={}

    n=10

    for num in uniques:

        if num==0:
            dictionary[num]=0
        else:
            dictionary[num]=n
            n=n+10

    for i in range(0,np.shape(seg)[0]):
        for j in range(0,np.shape(seg)[1]):
            for k in range(0,np.shape(seg)[2]):
                seg[i,j,k]=dictionary[seg[i,j,k]]

    return seg

def fix_seg(seg,dictionary,spacing_dictionary,list):


    new_seg = np.zeros(np.shape(seg))

    for i in range(0, np.shape(seg)[0]):
        for j in range(0, np.shape(seg)[1]):
            for k in range(0, np.shape(seg)[2]):
                if (dictionary[seg[i,j,k]]==list).any():
                    new_seg[i,j,k]=spacing_dictionary[seg[i,j,k]]

    return new_seg


def split_into_n_threads(n,seg,dictionary,spacing_dictionary,list):

    if n==1:
        return [[seg,dictionary,spacing_dictionary,list]]

    z=np.shape(seg)[0]

    size=int(z/n)

    args=[]

    stop=z

    for i in range(0,n-1):

        args.append([seg[i*size:(i+1)*(size)],dictionary,spacing_dictionary,list])


        stop=(i+1)*(size)-1


    if stop<z:

        args.append(([seg[stop:],dictionary,spacing_dictionary,list]))

    return args


def unify_threads(results):

    size=0
    for result in results:
        size+=np.shape(result)[0]

    array=np.zeros((size,np.shape(results[0])[1],np.shape(results[0])[2]),dtype=np.float32)

    zpos=0
    for result in results:
        array[zpos:np.shape(result)[0]+zpos]=result
        zpos=zpos+np.shape(result)[0]



    return array