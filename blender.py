import numpy as np
import math

import tifffile

def blend(arr_1,arr_2,dir,overlap,fac):
    #INPUT MUST BE ZXY

    '''
    DIRECTION INDICATES WHERE arr_2 is to be appended
    EX:
            "1-"

        arr2->arr1

            "1+"

        arr1<-arr2

            "2-"

        arr1
          ^
          |
        arr2

            "2+"

        arr2
          |
          \/
        arr1

    '''

    #write code to work in x+ direction and when x- is activated just switch arrays

    #make sure there are no issues
    arr_1=np.asarray(arr_1.copy())
    arr_2=np.asarray(arr_2.copy())


    #must be same size
    assert (np.shape(arr_1)==np.shape(arr_2)), "MISMATCH OF ARRAY SIZES"



    possible=np.asarray(["0+","0-","1+","1-","1+","1-"])

    assert (dir==possible).any, "INVALID DIRECTION"

    arr_b=np.asarray(9999999999)

    if(dir=="0+"):
        arr_1=np.einsum("czxy->cxzy",arr_1)
        arr_2=np.einsum("czxy->cxzy",arr_2)
        arr_b=xblend(arr_1,arr_2,overlap,fac)
        arr_b=np.einsum("cxzy->czxy",arr_b)
    if(dir=="0-"):
        arr_1 = np.einsum("czxy->cxzy", arr_1)
        arr_2 = np.einsum("czxy->cxzy", arr_2)
        arr_b = xblend(arr_2, arr_1, overlap,fac)
        arr_b = np.einsum("cxzy->czxy", arr_b)
    if(dir=="1+"):
        arr_b=xblend(arr_1,arr_2,overlap,fac)
    if(dir=="1-"):
        arr_b=xblend(arr_2,arr_1,overlap,fac)
    if(dir=="2+"):
        arr_1 = np.einsum("czxy->czyx", arr_1)
        arr_2 = np.einsum("czxy->czyx", arr_2)
        arr_b = xblend(arr_1, arr_2, overlap,fac)
        arr_b = np.einsum("czyx->czxy", arr_b)
    if(dir=="2-"):
        arr_1 = np.einsum("czxy->czyx", arr_1)
        arr_2 = np.einsum("czxy->czyx", arr_2)
        arr_b = xblend(arr_2, arr_1, overlap,fac)
        arr_b = np.einsum("czyx->czxy", arr_b)

    return arr_b


def xblend(arr_1,arr_2,overlap,fac):

    #define
    x_size = int((2 - overlap) * np.shape(arr_1)[2])
    overlap_size = int(np.shape(arr_1)[2] * overlap)

    # generate new array
    arr_b = np.zeros((np.shape(arr_1)[0],np.shape(arr_1)[1], int(x_size), np.shape(arr_1)[3]))

    # grab sections that overlap
    ovrlp_1 = arr_1[:,:, -int(overlap_size):, :].copy()

    ovrlp_2 = arr_2[:,:, 0:int(overlap_size), :].copy()



    #blending factor
    #1 gives linear
    #higher gives more arr2
    #lower gives more arr1



    #blending
    for x in range(0, int(overlap_size)):
        value=(x/overlap_size)**fac
        #value2=(x/overlap_size)**fac
        ovrlp_1[:, :, x, :] = np.multiply(1-value,ovrlp_1[:,:, x, :])
            #(((1/overlap_size)* x)**fac)*ovrlp_1[:,:, x, :]
        ovrlp_2[:, :, x, :] =np.multiply(value,ovrlp_2[:,:, x, :])
            #(1-((1/overlap_size)* x)**fac)*ovrlp_2[:,:, x, :]

    ovrlp_comb = np.add(ovrlp_1,ovrlp_2)

    # where does overlap begin index
    point_1 = np.shape(arr_1)[2] - overlap_size
    point_2 = np.shape(arr_1)[2]
    point_3 = overlap_size




    # add front 0->p1
    arr_b[:,:, 0:point_1, :] = arr_1[:,:, 0:point_1, :]
    # add middle p1->p2
    arr_b[:,:, point_1:point_2, :] = ovrlp_comb
    # add end p2->end
    arr_b[:,:, point_2:, :] = arr_2[:,:, point_3:, :]

    #tifffile.imsave("misc/preview",np.asarray(arr_b[0,:,-1,:],dtype=np.float32))

    return arr_b