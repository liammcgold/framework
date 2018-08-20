import time
import psutil
import numpy as np
import log
import os
import tifffile as tif
import assembler
import models
import waterz


best_model="malis_heavy_net/model500000"

best_model_type="heavy paralell UNET"

best_model_func=models.heavy_merged_u_net_make


def watershed(aff_vol,thresh=0.1,thresh_high=0.9999,thresh_low=0.0001):

    process = psutil.Process(os.getpid())
    mem_b = psutil.virtual_memory().used


    time_b = time.time()

    aff_vol = np.ascontiguousarray(aff_vol, dtype=np.float32)

    seg = waterz.agglomerate(aff_vol, thresholds=[thresh],aff_threshold_high=thresh_high,aff_threshold_low=thresh_low,discretize_queue=256)

    for segmentation in seg:
        seg=segmentation

    time_e = time.time()
    mem_a = psutil.virtual_memory().used

    log.log("watershed", np.shape(seg), time_e - time_b, mem_a - mem_b)

    return seg

def save_tiffs_no_gt(seg,aff_vol,raw_vol,gt_vol=None):

    print(np.shape(seg),np.shape(aff_vol),np.shape(raw_vol))

    for i in range(0, np.shape(seg)[0]):
        tif.imsave("predictions_tiffs/pred/pred%i.tiff" % i, np.asarray(seg[i], dtype=np.float32))  # ,photometric='rgb')
        #tif.imsave("predictions_tiffs/gt/gt%i.tiff" % i, np.asarray(gt_vol[i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph0/0aff%i.tiff" % i, np.asarray(aff_vol[0, i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph1/1aff%i.tiff" % i, np.asarray(aff_vol[1, i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph2/2aff%i.tiff" % i, np.asarray(aff_vol[2, i], dtype=np.float32))
        tif.imsave("predictions_tiffs/raw/raw%i" % i, np.asarray(raw_vol[i], dtype=np.float32))

def assemble_block_with_model_location(model_funct, model_file, raw_vol, overlap, blend_fac):
    time_b = time.time()
    mem_b = psutil.virtual_memory().used

    print("\nGrabing model...\n")

    model = model_funct()

    model.load_weights(model_file)

    def predict(raw_block):
        # temp=np.zeros((np.shape(raw_block)[aff_graph0]*2,np.shape(raw_block)[2]*2,np.shape(raw_block)[2]*2))

        # temp[int(np.shape(raw_block)[aff_graph0]/2):int(np.shape(raw_block)[aff_graph0]/2)+np.shape(raw_block)[aff_graph0],int(np.shape(raw_block)[1]/2):int(np.shape(raw_block)[1]/2)+np.shape(raw_block)[1],int(np.shape(raw_block)[2]/2):int(np.shape(raw_block)[2]/2)+np.shape(raw_block)[2]]=raw_block

        # temp=np.reshape(temp,(1,np.shape(temp)[aff_graph0],np.shape(temp)[1],np.shape(temp)[2],1))

        raw_block = np.reshape(raw_block, (1, 16, 128, 128, 1))

        aff = model.predict(raw_block)

        aff = np.einsum("bzxyc->bczxy", aff)

        aff = aff[0]

        return aff

    shape = model.input_shape[1:-1]

    builder = assembler.assembler(raw_vol, overlap, shape, predict, blend_fac)

    print("\nBuilding affinity block...\n")
    aff = builder.process()

    print("\nBlock building done.\n")

    time_e = time.time()
    mem_a = psutil.virtual_memory().used
    log.log("assemble", np.shape(raw_vol), time_e - time_b, mem_a - mem_b)

    return aff

def grab_from_tiff_stack(cords,tiff_folder="/media/user1/My4TBHD1/VCN_bin2_tiffs"):


    time_b = time.time()
    mem_b=psutil.virtual_memory().used

    ###cords should be in form [[zstart,zstop],[xstart,xstop],[ystart,ystop]]
    cords = np.asarray(cords)

    # load file lists
    raw_files = [1] * (len(os.listdir(tiff_folder)))
    n = 0
    for file in os.listdir(tiff_folder):

        if file.endswith(".tiff"):
            raw_files[n] = file
            n += 1

    raw_files = np.sort(raw_files)


    raw_tiff_array = np.empty((cords[0,1]-cords[0,0],cords[1,1]-cords[1,0],cords[2,1]-cords[2,0]), dtype=np.float32)


    for z in range(cords[0,0],cords[0,1]):
        print("Block %i of %i"%(z-cords[0,0],cords[0,1]-cords[0,0]))
        raw_tiff_array[z-cords[0,0]]=np.asarray(tif.imread(tiff_folder+"/"+raw_files[z]),dtype=np.float32)[cords[1,0]:cords[1,1],cords[2,0]:cords[2,1]]

    time_e = time.time()
    mem_a = psutil.virtual_memory().used
    log.log("loading", np.shape(raw_tiff_array), time_e - time_b, mem_a-mem_b)

    return raw_tiff_array

def process_block(cords,tiff_folder="/media/user1/My4TBHD1/VCN_bin2_tiffs",dump_folder="/media/user1/My4TBHD1/Dump",thresh=0.2,force_rebuild=[0,0,0]):


    time_b = time.time()
    mem_b = psutil.virtual_memory().used



    cords=np.asarray(cords)
    identifier="_z_%i_%i_x_%i_%i_y_%i_%i"%(cords[0,0],cords[0,1],cords[1,0],cords[1,1],cords[2,0],cords[2,1])


    if os.path.isfile(dump_folder+"/"+"raw"+identifier+".npy") and force_rebuild[0]==0:

        print("\nFound already built raw.\n")

        raw_vol=np.load(dump_folder+"/"+"raw"+identifier+".npy")

    else:

        print("\nGrabbing raw from tiffs...\n")

        raw_vol = grab_from_tiff_stack(cords, tiff_folder)
        np.save(dump_folder + "/" + "raw" + identifier, raw_vol)

        print("\nGot raw.\n")


    ###ENSURE RAW IS NORMALIZED IN RANGE [0,1]
    print("\nNormalizing raw array\n")
    min=np.min(raw_vol)
    max=np.max(raw_vol)
    mean=np.mean(raw_vol)
    a=.235
    b=.667

    if max>1 or min>.1:
        raw_vol=(raw_vol-min)/(max-min)
        # in range [0,1]
        # [0,1]->[a,b]
        raw_vol=(raw_vol*(b-a))+a
        #now in range a,b



    if os.path.isfile(dump_folder+"/"+"aff"+identifier+".npy") and force_rebuild[1]==0:

        print("\nFound already built aff.\n")

        aff_vol=np.load(dump_folder+"/"+"aff"+identifier+".npy")

    else:

        print("\nBuilding affinity block...\n")

        aff_vol = assemble_block_with_model_location(best_model_func, best_model, raw_vol, 0.75, 1)

        print("\nGot affinity block.\n")

        np.save(dump_folder + "/" + "aff" + identifier, aff_vol)


    if os.path.isfile(dump_folder+"/"+"seg"+identifier+str(thresh)+".npy") and force_rebuild[2]==0:

        print("\nFound already built seg.\n")

        seg=np.load(dump_folder+"/"+"seg"+identifier+".npy")

        save_tiffs_no_gt(seg, aff_vol, raw_vol)


    else:

        print("\nBuilding segmentation...\n")

        seg = watershed(aff_vol, thresh=thresh)


        seg=filter.top_n(seg,255)


        save_tiffs_no_gt(seg,aff_vol,raw_vol)

        print("\nSegmentation built..\n")

        np.save(dump_folder+"/"+"seg"+identifier+str(thresh),seg)

    time_e = time.time()
    process = psutil.Process(os.getpid())
    mem_a=process.memory_full_info().rss


    log.log("process", np.shape(raw_vol), time_e - time_b, mem_a)




    return seg
