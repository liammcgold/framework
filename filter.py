import numpy as np
import time

def filter(seg,thresh):

    new_seg=seg

    uniques=np.unique(seg)

    dictionary={}

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

    return new_seg


def top_n(seg,n):

    new_seg = np.zeros(np.shape(seg))

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


    for i in range(0, np.shape(seg)[0]):
        for j in range(0, np.shape(seg)[1]):
            for k in range(0, np.shape(seg)[2]):
                if (dictionary[seg[i,j,k]]==list).any():
                    new_seg[i,j,k]=dictionary[seg[i,j,k]]






    return new_seg


# x=np.asarray([[[0,1,2,3,3,4,5,6,6,6,7,8,8,8,8,8,9,10],[0,1,2,3,3,4,5,6,6,6,7,8,8,8,8,8,9,10]],
# [[0,1,2,3,3,4,5,6,6,6,7,8,8,8,8,8,9,10],[0,1,2,3,3,4,5,6,6,6,7,8,8,8,8,8,9,10]],
# [[0,1,2,3,3,4,5,6,6,6,7,8,8,8,8,8,9,10],[0,1,2,3,3,4,5,6,6,6,7,8,8,8,8,8,9,10]]])
#
# print(top_n(x,1))

