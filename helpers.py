import assembler
import function_sequence
import waterz
import tifffile as tif
import process
import custom_loss
import numpy as np
import os
import math
import tensorflow
import malis
import filter
import keras as k
import models
import log
import time
import psutil
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

raw = np.load("data/spir_raw.npy")
gt = np.load("data/spir_gt.npy")
aff = np.load("data/spir_aff.npy")

gpus=1#input("how many GPUS")


best_model="malis_heavy_net/model500000"

best_model_type="heavy paralell UNET"

best_model_func=models.heavy_merged_u_net_make

def assemble_block_with_model_location(model_funct, model_file, raw_vol, overlap, blend_fac):

        time_b = time.time()
        mem_b = psutil.virtual_memory().used

        print("\nGrabing model...\n")

        model=model_funct()

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


        shape=model.input_shape[1:-1]

        builder=assembler.assembler(raw_vol, overlap, shape, predict, blend_fac)

        print("\nBuilding affinity block...\n")
        aff=builder.process()

        print("\nBlock building done.\n")

        time_e = time.time()
        mem_a = psutil.virtual_memory().used
        log.log("assemble", np.shape(raw_vol), time_e - time_b, mem_a - mem_b)

        return aff

def assemble_block_with_model(model, raw_vol, overlap, blend_fac):


        def predict(raw_block):
            # temp=np.zeros((np.shape(raw_block)[aff_graph0]*2,np.shape(raw_block)[2]*2,np.shape(raw_block)[2]*2))

            # temp[int(np.shape(raw_block)[aff_graph0]/2):int(np.shape(raw_block)[aff_graph0]/2)+np.shape(raw_block)[aff_graph0],int(np.shape(raw_block)[1]/2):int(np.shape(raw_block)[1]/2)+np.shape(raw_block)[1],int(np.shape(raw_block)[2]/2):int(np.shape(raw_block)[2]/2)+np.shape(raw_block)[2]]=raw_block

            # temp=np.reshape(temp,(1,np.shape(temp)[aff_graph0],np.shape(temp)[1],np.shape(temp)[2],1))

            raw_block = np.reshape(raw_block, (1, 16, 128, 128, 1))

            aff = model.predict(raw_block)

            aff = np.einsum("bzxyc->bczxy", aff)

            aff = aff[0]

            return aff


        shape=model.input_shape[1:-1]


        builder=assembler.assembler(raw_vol, overlap, shape, predict, blend_fac)

        print("\nBuilding affinity block...\n")
        aff=builder.process()

        print("\nBlock building done.\n")

        return aff

def predict_and_watershed_on_list_of_models(model_list,raw_vol=raw[0:32], gt_vol=gt[0:32],overlap=.5,blend_fac=1,metric="choose"):


    segs=[]
    metrics=[]

    for model in model_list:

        aff=assemble_block_with_model(model,raw_vol,overlap,blend_fac)

        seg, metric_slice=watershed_sweep(aff,gt_vol,metric=metric,return_all=True)


        for seg_sample in seg:
            segs.append(seg_sample)

        for metric_sample in metric_slice:
            metrics.append(metric_sample)

    return segs,metrics

def predict_and_watershed_on_vol_with_gt_conf_and_save_tiffs(model_funct,model_file,raw_vol=raw[0:32,:-50,:-50], gt_vol=gt[0:32,:-50,:-50],overlap=.5,blend_fac=1,metric="both"):

    print("Using trained model to generate affinity block...")

    aff=assemble_block_with_model_location(model_funct,model_file,raw_vol,overlap,blend_fac)

    seg=watershed_sweep(aff,gt_vol,metric=metric)

    for i in range(0, np.shape(seg)[0]):
        tif.imsave("predictions_tiffs/pred/pred%i.tiff" % i, np.asarray(seg[i], dtype=np.float32))  # ,photometric='rgb')
        tif.imsave("predictions_tiffs/gt/gt%i.tiff" % i, np.asarray(gt_vol[i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph0/0aff%i.tiff" % i, np.asarray(aff[0, i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph1/1aff%i.tiff" % i, np.asarray(aff[1, i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph2/2aff%i.tiff" % i, np.asarray(aff[2, i], dtype=np.float32))

def predict_and_watershed_on_vol_with_gt_conf_and_save_tiffs_with_model(model,raw_vol=raw[0:32], gt_vol=gt[0:32],overlap=.5,blend_fac=1,metric="both"):

    print("Using trained model to generate affinity block...")

    aff=np.asarray(assemble_block_with_model(model,raw_vol,overlap,blend_fac))

    seg=watershed_sweep(aff,np.asarray(gt_vol),metric=metric)

    for i in range(0, np.shape(seg)[0]):
        tif.imsave("predictions_tiffs/pred/pred%i.tiff" % i, np.asarray(seg[i], dtype=np.float32))  # ,photometric='rgb')
        tif.imsave("predictions_tiffs/gt/gt%i.tiff" % i, np.asarray(gt_vol[i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph0/0aff%i.tiff" % i, np.asarray(aff[0, i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph1/1aff%i.tiff" % i, np.asarray(aff[1, i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph2/2aff%i.tiff" % i, np.asarray(aff[2, i], dtype=np.float32))

def predict_and_watershed_on_vol_with_gt_conf_and_save_tiffs_single_thresh(model_funct,model_file,thresh,thresh_high=0.9999,thresh_low=0.0001,raw_vol=raw[0:32,:,:-50], gt_vol=gt[0:32,:,:-50],overlap=.5,blend_fac=1):

    print("Using trained model to generate affinity block...")

    aff=assemble_block_with_model_location(model_funct,model_file,raw_vol,overlap,blend_fac)

    seg=__watershed_return_metrics_single(aff,gt_vol,thresh,metric="both",no_metric_flag=1,thresh_high=thresh_high,thresh_low=thresh_low)

    for i in range(0, np.shape(seg)[0]):
        tif.imsave("predictions_tiffs/pred/pred%i.tiff" % i, np.asarray(seg[i], dtype=np.float32))  # ,photometric='rgb')
        tif.imsave("predictions_tiffs/gt/gt%i.tiff" % i, np.asarray(gt_vol[i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph0/0aff%i.tiff" % i, np.asarray(aff[0, i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph1/1aff%i.tiff" % i, np.asarray(aff[1, i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph2/2aff%i.tiff" % i, np.asarray(aff[2, i], dtype=np.float32))

def watershed_on_vol_with_gt_conf_and_save_tiffs_single_thresh(aff_vol,gt_vol,thresh,thresh_high=0.9999,thresh_low=0.0001):


    seg=__watershed_return_metrics_single(aff_vol,gt_vol,thresh,metric="both",no_metric_flag=1,thresh_high=thresh_high,thresh_low=thresh_low)

    seg=filter.top_n(seg,2000)


    for i in range(0, np.shape(seg)[0]):
        tif.imsave("predictions_tiffs/pred/pred%i.tiff" % i, np.asarray(seg[i], dtype=np.float32))  # ,photometric='rgb')
        tif.imsave("predictions_tiffs/gt/gt%i.tiff" % i, np.asarray(gt_vol[i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph0/0aff%i.tiff" % i, np.asarray(aff[0, i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph1/1aff%i.tiff" % i, np.asarray(aff[1, i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph2/2aff%i.tiff" % i, np.asarray(aff[2, i], dtype=np.float32))
    return seg

def watershed_on_vol_and_save_tiffs_single_thresh(aff_vol,gt_vol,thresh,thresh_high=0.9999,thresh_low=0.0001):


    seg=watershed(aff_vol,thresh=thresh,thresh_high=thresh_high,thresh_low=thresh_low)


    for i in range(0, np.shape(seg)[0]):
        tif.imsave("predictions_tiffs/pred/pred%i.tiff" % i, np.asarray(seg[i], dtype=np.float32))  # ,photometric='rgb')
        tif.imsave("predictions_tiffs/gt/gt%i.tiff" % i, np.asarray(gt_vol[i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph0/0aff%i.tiff" % i, np.asarray(aff[0, i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph1/1aff%i.tiff" % i, np.asarray(aff[1, i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph2/2aff%i.tiff" % i, np.asarray(aff[2, i], dtype=np.float32))
    return seg

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

def __watershed_return_metrics_single(aff_vol, gt_vol, thresh, metric="both",no_metric_flag=0,thresh_high=0.9999,thresh_low=0.0001):





    aff_vol=np.ascontiguousarray(aff_vol, dtype=np.float32)
    gt_vol=np.ascontiguousarray(gt_vol, dtype=np.uint32)

    seg = waterz.agglomerate(aff_vol, thresholds=[thresh], gt=gt_vol,aff_threshold_high=thresh_high,aff_threshold_low=thresh_low)

    for segmentation in seg:
        seg=segmentation

    if no_metric_flag==0:
        metrics=seg[1]
        if str(metric)=="both":
            metric = metrics['V_Rand_split']+metrics["V_Rand_merge"]

        elif str(metric)=="split":

            metric = metrics['V_Rand_split']

        elif str(metric)=="merge":

            metric = metrics['V_Rand_merge']
        else:
            metric=metrics

        return seg[0],metric
    else:
        return seg[0]

def __watershed_return_metrics_list(aff_vol, gt_vol, thresh, metric="both",thresh_high=0.9999,thresh_low=0.0001):

    aff_vol=np.ascontiguousarray(aff_vol, dtype=np.float32)
    gt_vol=np.ascontiguousarray(gt_vol, dtype=np.uint32)


    if str(metric)=="choose":
        metrics=np.zeros((np.shape(thresh)[0],2))

    else:
        metrics=np.zeros(np.shape(thresh))


    segmentations=np.zeros(np.shape(thresh),dtype=np.object)

    seg = waterz.agglomerate(aff_vol, thresholds=thresh, gt=gt_vol,aff_threshold_high=thresh_high,aff_threshold_low=thresh_low)

    n=0

    for segmentation in seg:

        segmentations[n]=segmentation[0]


        dictionary=segmentation[1]

        if str(metric) == "both":

            metrics[n] = dictionary['V_Rand_split'] + dictionary["V_Rand_merge"]

        elif str(metric) == "split":
            metrics[n] = dictionary['V_Rand_split']

        elif str(metric) == "merge":
            metrics[n] = dictionary['V_Rand_merge']

        elif str(metric) == "choose":
            metrics[n,0] = dictionary['V_Rand_split']
            metrics[n,1] = dictionary['V_Rand_merge']

        else:
            assert 1 == 0, "Bad metric string"

        n+=1



    return segmentations,metrics

def watershed_return_metrics(aff_vol, gt_vol, thresh, metric="both"):

    if np.shape(thresh)[0]>1:
        return __watershed_return_metrics_list(aff_vol, gt_vol, thresh, metric=metric)
    else:
        return __watershed_return_metrics_single(aff_vol, gt_vol, thresh, metric=metric)

def watershed_sweep(aff_vol, gt_vol, metric="both",return_all=False):

    assert np.shape(aff_vol)[1:] == np.shape(gt_vol), "Shape mismatch"

    values=[.1,.2,.3,.4,.5,.6,.7,.8,.9]

    if return_all==False:

        if metric!="choose":

            seg=np.zeros(np.shape(values)[0],dtype=object)
            metrics=np.zeros(np.shape(values)[0],dtype=object)


            seg[:],metrics[:]=watershed_return_metrics(aff_vol, gt_vol, values, metric=metric)


            index=np.where(metrics == np.max(metrics))

            index=index[0]

            print("BEST PERFORMANCE WAS ="+str(np.max(metrics)))

            return seg[int(index)]

        else:

            seg = np.zeros(np.shape(values)[0], dtype=object)
            metrics = np.zeros((np.shape(values)[0],2), dtype=object)

            seg[:], metrics[:] = watershed_return_metrics(aff_vol, gt_vol, values, metric=metric)

            print("\nChoose segmentation from below:")

            n=0
            for metric in metrics:

                print("\tSegmentation %i: "%n)
                print("\t\tSplit: "+str(metric[0]))
                print("\t\tMerge: "+str(metric[1]))
                n+=1

            index=input("\n\nWhich model?")

            return seg[int(index)]

    else:

        seg = np.zeros(np.shape(values)[0], dtype=object)

        metrics = np.zeros((np.shape(values)[0],2), dtype=object)

        seg[:], metrics[:] = watershed_return_metrics(aff_vol, gt_vol, values, metric=metric)

        return seg, metrics

def predict_affins_and_save(model_funct,model_file,raw_vol,overlap,save_loc,shape=[16,128,128]):

    model=model_funct()

    model.load_weights(model_file)

    def predict(raw_block):
        # temp=np.zeros((np.shape(raw_block)[aff_graph0]*2,np.shape(raw_block)[2]*2,np.shape(raw_block)[2]*2))

            # temp[int(np.shape(raw_block)[aff_graph0]/2):int(np.shape(raw_block)[aff_graph0]/2)+np.shape(raw_block)[aff_graph0],int(np.shape(raw_block)[1]/2):int(np.shape(raw_block)[1]/2)+np.shape(raw_block)[1],int(np.shape(raw_block)[2]/2):int(      np.shape( raw_block)[2]/2)+np.shape(raw_block)[2]]=raw_block

        # temp=np.reshape(temp,(1,np.shape(temp)[aff_graph0],np.shape(temp)[1],np.shape(temp)[2],1))

        raw_block = np.reshape(raw_block, (1, 16, 128, 128, 1))

        aff = model.predict(raw_block)

        aff = np.einsum("bzxyc->bczxy", aff)

        aff = aff[0]

        return aff


    tag=model_file.split("/")[-1]

    builder = assembler.assembler(raw_vol, overlap, shape, predict,1)

    print("\nBuilding affinity block...\n")
    aff = builder.process()

    print("\nBlock building done.\n")

    np.save(save_loc+"/predicted_affinities-"+tag,aff)

    return aff

def is_stable(list):

    differentials=[]

    if len(list)<30:
        return False

    for i in range(0,np.shape(list)[0]-1):
        differential=list[i+1]-list[i]
        differentials.append(differential)


    diff=list[0]-list[-1]
    standard_step = diff / len(differentials)

    average=np.average(list)

    average_dif=np.average(differentials)
    last_5_average_dif=np.average(differentials[-5:])

    std=np.std(list)


    if len(list)>20 and std<.01*average:
        return True


    if abs(last_5_average_dif)<abs(.15*average_dif):
        return True
    elif abs(last_5_average_dif)<abs((diff/len(differentials))*.1):
        return True
    else:
        return False

def test_for_bad_convergence(process):

    print("\tTesting for bad convergence...")

    pred=process.model.predict(process.conf_raw)

    std=np.std(pred)

    if (std < 0.01 and process.iteration > 10000): #or (std < 0.001 and process.iteration > 1000):
        print("\n\nBAD CONVERGENCE\n")
        return False

def update_LR(process):

    print("\n\tUpdating LR...")

    list=process.validation_loss_list

    if is_stable(list):
        print("\n\tLoss stablaized,lowering LR.")
        process.decrement_lr()
        process.reset_validation_loss_data()
        return


    print("\n\tLoss still unstable.\n")

def train_from_scratch(saving_loc,loss=None,initial_lr=0.0025,gpus=1,model_type="heavy paralell UNET",image_interval=10,check_interval=100,validation_interval=200):

    if loss==None:
        l = custom_loss.loss()

        weights = np.zeros((3, 2))

        weights[0] = [2.6960856, 0.61383891]
        weights[1] = [4.05724285, 0.57027915]
        weights[2] = [4.09752934, 0.56949214]

        weights = np.asarray(weights)

        l.set_weight(weights)

        loss=l.weighted_cross



    n = 544

    proc = process.process(loss,
                           raw,
                           gt,
                           aff,
                           model_type=model_type,
                           precision="half",
                           save_loc=saving_loc,
                           saving_sched=[[0, 100], [1000, 10000], [100000, 20000]],
                           image_loc="tiffs/",
                           image_interval=image_interval,
                           check_interval=check_interval,
                           conf_coordinates=[[200, 216], [200, 328], [200, 328]],
                           learning_rate=initial_lr,
                           validation_frac=.2,
                           validation_interval=validation_interval,
                           check_function=checks,
                           gpus=gpus
                           )

    flag = proc.train(500000)
    while (not flag):
        proc.iteration = 0
        proc.learning_rate = proc.learning_rate * .1
        print("\n\tNEW LR = %f"%proc.learning_rate)
        flag = proc.train(500000)

    print("training complete using LR=" + str(proc.learning_rate))

def checks(process):

    print("\n--------CHECKS--------\n")

    if test_for_bad_convergence(process)==False:
        return False
    else:
        print("\n\tNo bad convergence.")
        update_LR(process)
    print("----------------------")

def run_process_with_checks_saved_model(iteration,
                                        raw_vol=raw,
                                        gt_vol=gt,
                                        aff_vol=aff,
                                        initial_lr=0.00025,
                                        check_function=checks,
                                        validation_interval=200,
                                        model_type="heavy paralell UNET",
                                        save_loc="malis_heavy_net/",
                                        check_interval=100,
                                        saving_schedule=[[0, 100], [1000, 10000]],
                                        loss=None,
                                        image_interval=10,
                                        gpus=1,
                                        random_contrast=False):
    if loss==None:

        loss = custom_loss.loss()

        weights = np.zeros((3, 2))

        weights[0] = [2.6960856, 0.61383891]
        weights[1] = [4.05724285, 0.57027915]
        weights[2] = [4.09752934, 0.56949214]

        weights = np.asarray(weights)

        loss.set_weight(weights)

        loss=loss.weighted_cross


    n = 544

    proc = process.process(loss,
                           raw_vol,
                           gt_vol,
                           aff_vol,
                           model_type=model_type,
                           precision="half",
                           save_loc=save_loc,
                           saving_sched=saving_schedule,
                           image_loc="tiffs/",
                           check_interval=check_interval,
                           image_interval=image_interval,
                           conf_coordinates=[[200, 216], [200, 328], [200, 328]],
                           learning_rate=initial_lr,
                           validation_frac=.2,
                           validation_interval=validation_interval,
                           check_function=check_function,
                           pickup_file="malis_heavy_net/model%i"%iteration,
                           pickup_iteration=iteration,
                           gpus=gpus,
                           random_contrast=random_contrast,
                           )
    try:
        flag = proc.train(500000)
        while (not flag):
            proc.iteration = 0
            proc.learning_rate = proc.learning_rate * .1
            flag = proc.train(500000)
    except KeyboardInterrupt:
        raw_vol=raw_vol[0:32]
        gt_vol=gt_vol[0:32]
        predict_and_watershed_on_vol_with_gt_conf_and_save_tiffs_with_model(proc.model,raw_vol,gt_vol,.5,1)



    print("training complete using LR=" + str(proc.learning_rate))

def validate_with_model(model,loss,validation_frac):

    process.process(loss)

def save_segmentation_tifs(gt_vol,seg):

    for i in range(0, np.shape(seg)[0]):
        tif.imsave("predictions_tiffs/pred/pred%i.tiff" % i, np.asarray(seg[i], dtype=np.float32))  # ,photometric='rgb')
        tif.imsave("predictions_tiffs/gt/gt%i.tiff" % i, np.asarray(gt_vol[i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph0/0aff%i.tiff" % i, np.asarray(aff[0, i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph1/1aff%i.tiff" % i, np.asarray(aff[1, i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph2/2aff%i.tiff" % i, np.asarray(aff[2, i], dtype=np.float32))

def find_best_segmentation(model_funct,folder,validation_frac,metric="choose",raw_vol=raw,gt_vol=gt,aff_vol=aff,top_n_valid=3,gpus=1):

    l = custom_loss.loss()

    weights = np.zeros((3, 2))

    weights[0] = [2.6960856, 0.61383891]
    weights[1] = [4.05724285, 0.57027915]
    weights[2] = [4.09752934, 0.56949214]

    models=[]
    model_keys=[]

    print("\nGrabbing models...\n")
    for file in os.listdir(folder):

        model=model_funct(verbose=0)

        try:
            model.load_weights(folder+"/"+file)
            models.append(model)
            model_keys.append(file)
        except OSError:
            print("\n%s is invalid\n"%file)
            continue

    print("\nLoaded %i models.\n"%len(models))
    for key in model_keys:
        print("\t"+key)

    proc=process.process(l.weighted_cross,raw_vol,gt_vol,aff_vol,model=models[0],validation_frac=validation_frac,gpus=gpus)

    valid_loss=[]

    print("\nGetting validation loss for all models...\n")
    for model in models:

        proc.model=model

        valid_loss.append(proc.calc_validation_loss())

    top_models=[]
    top_model_keys=[]

    top_indexs = np.asarray(valid_loss).argsort()[-top_n_valid:][::-1]


    for index in top_indexs:

        top_models.append(models[index])
        top_model_keys.append(model_keys[index])

    print("\nFound top %i models.\n"%top_n_valid)

    for key in top_model_keys:
        print("\t%s"%key)

    print("\nWatershed sweep...\n")
    segs,metrics=predict_and_watershed_on_list_of_models(top_models,metric=metric)


    if metric != "choose":

        index = np.where(metrics == np.max(metrics))

        index = index[0]

        print("BEST PERFORMANCE WAS =" + str(np.max(metrics)))

        seg=segs[int(index)]

    else:

        n = 0
        for metric in metrics:
            print("\tSegmentation %i: " % n)
            print("\t\tSplit: " + str(metric[0]))
            print("\t\tMerge: " + str(metric[1]))
            n += 1

        index = input("\n\nWhich model?")

        seg=segs[int(index)]

    model_key=top_model_keys[math.floor((int(index)/len(segs))*len(top_models))]

    print("Model %s is the choice"%model_key)

    save_segmentation_tifs(gt_vol,seg)

def generate_aff_graph(gt,save_location=None):

    seg=malis.seg_to_affgraph(gt)

def get_layer_output_grad(model, inputs, outputs, layer=-1):
    """ Gets gradient a layer output for given inputs and outputs"""
    grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = k.backend.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad

def tiffs_to_npy(tiff_folder,output_file):
    # load file lists
    raw_files = [1] * (len(os.listdir(tiff_folder)))
    n = 0
    for file in os.listdir(tiff_folder):

        if file.endswith(".tiff"):
            raw_files[n] = file
            n += 1

    raw_files = np.sort(raw_files)

    sample_ar=np.asarray(tif.imread(tiff_folder + "/" + raw_files[0]))

    numbers=len(raw_files)*np.shape(sample_ar)[0]*np.shape(sample_ar)[1]

    bit_percision=16

    memory=numbers*bit_percision/8/1000/1000/1000

    print("array is %0.4f Gigabytes"%memory)

    raw_tiff_array = np.empty((len(raw_files),np.shape(sample_ar)[0],np.shape(sample_ar)[1]),dtype=np.float16)

    n = 0
    for file in raw_files:
        print(tiff_folder + "/" + file)
        raw_tiff_array= np.append(raw_tiff_array,[np.asarray(tif.imread(tiff_folder + "/" + file), dtype=np.float32)],axis=0)
        n += 1

    raw_tiff_array = np.asarray(raw_tiff_array, dtype=np.float32)
    raw_tiff_array = raw_tiff_array * (1 / 255)

    np.save(output_file,raw_tiff_array)

def save_tiffs_no_gt(seg,aff_vol,raw_vol,gt_vol=None):

    print(np.shape(seg),np.shape(aff_vol),np.shape(raw_vol))

    for i in range(0, np.shape(seg)[0]):
        tif.imsave("predictions_tiffs/pred/pred%i.tiff" % i, np.asarray(seg[i], dtype=np.float32))  # ,photometric='rgb')
        #tif.imsave("predictions_tiffs/gt/gt%i.tiff" % i, np.asarray(gt_vol[i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph0/0aff%i.tiff" % i, np.asarray(aff_vol[0, i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph1/1aff%i.tiff" % i, np.asarray(aff_vol[1, i], dtype=np.float32))
        tif.imsave("predictions_tiffs/aff_graph2/2aff%i.tiff" % i, np.asarray(aff_vol[2, i], dtype=np.float32))
        tif.imsave("predictions_tiffs/raw/raw%i" % i, np.asarray(raw_vol[i], dtype=np.float32))

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

        seg = watershed(aff_vol[:,75:-75,75:-75,75:-75], thresh=thresh)


        seg=filter.top_n(seg,255)


        save_tiffs_no_gt(seg,aff_vol[:,75:-75,75:-75,75:-75],raw_vol[75:-75,75:-75,75:-75])

        print("\nSegmentation built..\n")

        np.save(dump_folder+"/"+"seg"+identifier+str(thresh),seg)

    time_e = time.time()
    process = psutil.Process(os.getpid())
    mem_a=process.memory_full_info().rss


    log.log("process", np.shape(raw_vol), time_e - time_b, mem_a)




    return seg
