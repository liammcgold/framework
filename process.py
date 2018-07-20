import models
import numpy as np
import keras as k
import math
import random_provider
import time
import tensorflow as tf
import tifffile
import assembler


class process:

    def __init__(self,
                 loss,
                 raw,
                 gt,
                 aff,
                 model_type=None,
                 model=None,
                 loss_sched=None,
                 saving_sched=None,
                 pickup_iteration=0,
                 pickup_file=None,
                 save_loc="",
                 validation_frac=None,
                 learning_rate=.0000025,
                 precision=None,
                 image_interval=10,
                 check_interval=100,
                 validation_interval=100,
                 image_loc=None,
                 conf_coordinates=None,
                 check_function=None,gpus=1):

        #all volumes should be passed with shape "CZXY"

        #intialize model and loss
        self.__initialize_model(model_type,model)
        self.__initialize_loss(loss,loss_sched)

        #initialize source arrays
        self.__initialize_arrays(raw,gt,aff)

        #initialize precision
        self.precision=precision
        #self.__initialize_precision()

        #initialize validation set
        self.__initialize_validation_set(validation_frac)

        #initialize provider that will give slices
        self.provider=random_provider.provider(self.raw,self.aff,self.gt,self.input_shape[1:4])

        #initialize iteration and load file if necesary
        self.__initialize_iteration(pickup_iteration,pickup_file)

        #initialize optimizer
        self.__initialize_optimizer(learning_rate)

        #initialize saving
        self.__initialize_saver(saving_sched,save_loc)

        #initialize image saving
        self.__initialize_image_saving(image_interval, image_loc)

        #initialize conf set
        self.__initialize_conf_set(conf_coordinates)

        self.check_function=check_function

        self.validation_loss_list=[]

        self.validation_interval=validation_interval

        self.check_interval=check_interval

        self.image_interval=image_interval

        self.gpus=gpus

    def __initialize_conf_set(self,conf_coordinates):

        assert (self.image_loc==None and conf_coordinates==None) or (self.image_loc!=None and conf_coordinates!=None) or (self.image_loc!=None and conf_coordinates==None), "CONF COORDINATES ARE NOT USED IF NO IMAGE SAVE LOC IS SPECIFIED"

        if(conf_coordinates==None):
            conf_coordinates=[[0,self.input_shape[1]],[0,self.input_shape[2]],[0,self.input_shape[3]]]

        conf_coordinates = np.asarray(conf_coordinates)



        assert (conf_coordinates[0,1]-conf_coordinates[0,0])==self.input_shape[1], "Z VALUE FOR CONF COORDINATES IS BAD"
        assert conf_coordinates[1,1]-conf_coordinates[1,0]==self.input_shape[2], "X VALUE FOR CONF COORDINATES IS BAD"
        assert conf_coordinates[2,1]-conf_coordinates[2,0]==self.input_shape[3], "Y VALUE FOR CONF COORDINATES IS BAD"




        conf_raw = np.zeros((1, 1, self.input_shape[1], self.input_shape[2], self.input_shape[3]))
        conf_raw[0,0]=self.raw[conf_coordinates[0,0]:conf_coordinates[0,1],conf_coordinates[1,0]:conf_coordinates[1,1],conf_coordinates[2,0]:conf_coordinates[2,1]]
        conf_raw = np.einsum("bczxy->bzxyc", conf_raw)
        self.conf_raw=conf_raw


        conf_aff = np.zeros((1, 3, self.input_shape[1], self.input_shape[2], self.input_shape[3]))
        conf_aff[0] = self.aff[:,conf_coordinates[0,0]:conf_coordinates[0,1],conf_coordinates[1,0]:conf_coordinates[1,1],conf_coordinates[2,0]:conf_coordinates[2,1]]
        conf_aff = np.einsum("bczxy->bzxyc", conf_aff)
        self.conf_aff=conf_aff



        self.conf_gt=self.gt[conf_coordinates[0,0]:conf_coordinates[0,1],conf_coordinates[1,0]:conf_coordinates[1,1],conf_coordinates[2,0]:conf_coordinates[2,1]]


    def __initialize_image_saving(self,image_interval,image_loc):

        if image_loc==None:

            self.check_interval=999999999999999999
            self.image_loc=image_loc

        else:
            self.check_interval=image_interval
            self.image_loc=image_loc


    def __initialize_saver(self,sched,loc):
        '''

        schedule should be in form:
                                    [[0,how many iterations between saves],[itteration that new increment takes effect, iter. between saves]]

        for example:
                    [[0,10],[100,100]]
            this will save every 10 iterations until the 100th iteration, then it will save every 100

        '''



        self.saving_schedule=np.asarray(sched)
        self.save_loc=loc


    def __initialize_optimizer(self,learning_rate):
        self.learning_rate = learning_rate
        self.optimizer = k.optimizers.Adam(lr=self.learning_rate)


    def __initialize_iteration(self,pickup_iteration,pickup_file):
        self.iteration = pickup_iteration

        if (self.iteration != 0):
            assert pickup_file != None, "IF PICKING UP AT NON ZERO POINT MUST PROVIDE FILE TO LOAD"

        if (pickup_file != None):
            self.model.load_weights(pickup_file)


    def __initialize_arrays(self,raw,gt,aff):
        self.raw = np.asarray(raw)
        self.gt = np.asarray(gt)
        self.aff = np.asarray(aff)

        assert self.raw.shape == self.gt.shape and self.gt.shape == self.aff.shape[ 1:], "RAW, GT AND AFF SHAPES DO NOT AGREE"

        assert (self.aff.shape[0] == 3), "AFFINITIES MUST BE PASSED IN SHAPE CZXY"


    def __initialize_validation_set(self,validation_frac):

        #cut by percentage in z dir

        if(validation_frac!=None):

            self.validation=True

            z_dim=self.raw.shape[0]

            z_slice_size=z_dim*validation_frac

            z_slice_size=math.ceil(z_slice_size/self.input_shape[1])*self.input_shape[1]

            self.validation_raw =self.raw[-z_slice_size:]
            self.validation_gt=self.gt[-z_slice_size:]
            self.validation_aff=self.aff[:,-z_slice_size:]

            self.raw=self.raw[:-z_slice_size]
            self.gt=self.gt[:-z_slice_size]
            self.aff=self.aff[:,:-z_slice_size]
        else:
            self.validation=False

    def __initialize_model(self,model_type,model):
        # must specify either a type or pass a model
        assert (model_type != None or model != None), "MUST SPECIFY EITHER TYPE TO LOAD OR PASS MODEL OBJECT"

        # cannot have two models

        assert not (model_type != None and model != None), "DO NOT SPECIFY A MODEL TYPE AND PASS A MODEL"

        model_dict = {"small kernel UNET": models.small_kernel_u_net_make, "large kernel UNET": models.large_kernel_make,
                      "parallel kernel UNET": models.merged_u_net_make(), "heavy paralell UNET": models.heavy_merged_u_net_make}

        # if string is given build specified model
        if model_type != None:
            assert type(model_type) == str, "MUST PASS STRING FOR TYPE"
            assert model_type in model_dict, "TYPE MUST MATCH A TYPE IN TYPE LIST"

            self.model = model_dict[model_type](verbose=0)

        # if model passed initialize it
        if model != None:
            assert type(model) == k.engine.training.Model, "PASSED INVALID MODEL"

            self.model = model

        self.input_shape=[1,int(self.model.input.shape[1]),int(self.model.input.shape[2]),int(self.model.input.shape[3]),int(self.model.input.shape[4])]
        self.output_shape=[1,int(self.model.output.shape[1]),int(self.model.output.shape[2]),int(self.model.output.shape[3]),int(self.model.output.shape[4])]


    def __initialize_loss(self,loss,loss_sched):
        self.loss = loss

        # check if loss is list or method
        if type(self.loss) == list:
            for method in self.loss:
                assert str(type(method)) == "<class 'method'>", "LIST CONTAINED NON METHODS, MUST PASS LIST OF METHODS"
            self.loss_schedule_flag = 1


        elif str(type(self.loss)) == "<class 'method'>":
            self.loss_schedule_flag = 0

        assert self.loss_schedule_flag in [0, 1], "LOSS MUST BE METHOD OR LIST OF METHODS"


        if self.loss_schedule_flag == 1:
            assert loss_sched != None, "PASSED LIST WITH NO SCHEDULE"

            #check validity of schedule
            last_entry=0
            for entry in loss_sched:
                assert entry[0]>=0,"PASSED BAD ENTRY FOR DICTIONARY KEY, PASS IN FORM dict={0:index of loss to use,index to change losses:new loss index...}"



            for entry in loss_sched:
                assert entry[1]<len(loss), "INDEX GIVEN FOR LOSS FUNCTION IS OUT OF LISTS RANGE, PASS IN FORM dict={0:index of loss to use,index to change losses:new loss index...}"

            self.loss_schedule=np.asarray(loss_sched)


    def train(self,iterations):

        if self.iteration==0:
            self.validation_loss_list=[]

        try:
            time_s0 = time.time()
            while self.iteration<=iterations:


                time_s1=0

                print("\nGrabbing loss...\n")

                model_loss=self.__get_loss()

                #self.model.compile(loss=model_loss,optimizer=self.optimizer,metrics=['accuracy'])
                self.__compile_model(model_loss)


                event_index=self.__get_critical_index(iterations)

                save_index=self.__get_save_index()

                while self.iteration<=event_index:




                    #TIME STUFF, DO NOT PUT ANYTHING ABOVE OR WILL GET INACCURATE TIMES
                    if time_s1==0:
                        time_s1=time_s0
                    else:
                        time_s1=time.time()

                    print("--------ITERATION %i--------"%self.iteration)

                    if self.iteration%self.image_interval==0:
                        self.save_tiffs()


                    if self.iteration%self.validation_interval==0:
                        if self.validation:
                            self.validation_loss_list.append(self.calc_validation_loss())

                    if self.iteration%self.check_interval==0:
                        if(self.check_function!=None):
                            if(self.check_function(self)==False):
                                return False

                    self.__train_iteration()


                    if self.iteration==save_index:
                        self.save_model()
                        self.iteration += 1
                        save_index=self.__get_save_index()
                    else:
                        self.iteration += 1


                    print("Time: %ims"%int(1000*(time.time()-time_s1)))
                    print("----------------------------\n\n")

                # TIME STUFF, DO NOT PUT ANYTHING below OR WILL GET INACCURATE TIMES
                time_s0=time.time()
        except KeyboardInterrupt:
            print("SAVING MODEL EARLY")
            self.save_model()
            print(self.learning_rate)
            raise KeyboardInterrupt
        return True

    def __compile_model(self,loss):

        if self.gpus>1:
            self.model = k.utils.multi_gpu_model(self.model, gpus=self.gpus)

        self.model.compile(loss=loss,optimizer=self.optimizer,metrics=['accuracy'])


    def __get_save_index(self):

        step=self.saving_schedule[-1,1]

        n=0
        for i in range(0,np.shape(self.saving_schedule)[0]):

            if self.iteration<self.saving_schedule[i,0]:
                step=self.saving_schedule[i-1,1]
                break

        return (math.floor(self.iteration//step)+1)*step


    def save_tiffs(self):

        if self.image_loc==None:
            return



        else:
            print("\nSaving Tiffs...")
            pred = self.model.predict(self.conf_raw)

            gradients=get_layer_output_grad(self.model,self.conf_raw,self.conf_aff)


            for j in range(0, 16):
                tifffile.imsave("training_tiffs/pred0/0predicted_affins%i" % j,
                                np.asarray(pred, dtype=np.float32)[0, j, :, :, 0])
                tifffile.imsave("training_tiffs/pred1/1predicted_affins%i" % j,
                                np.asarray(pred, dtype=np.float32)[0, j, :, :, 1])
                tifffile.imsave("training_tiffs/pred2/2predicted_affins%i" % j,
                                np.asarray(pred, dtype=np.float32)[0, j, :, :, 2])
                tifffile.imsave("training_tiffs/act0/0actual_affins%i" % j,
                                np.asarray(self.conf_aff, dtype=np.float32)[0, j, :, :, 0])
                tifffile.imsave("training_tiffs/act1/1actual_affins%i" % j,
                                np.asarray(self.conf_aff, dtype=np.float32)[0, j, :, :, 1])
                tifffile.imsave("training_tiffs/act2/2actual_affins%i" % j,
                                np.asarray(self.conf_aff, dtype=np.float32)[0, j, :, :, 2])
                tifffile.imsave("training_tiffs/raw/raw%i" % j,
                                np.asarray(self.conf_raw, dtype=np.float32)[0, j, :, :, 0])
                tifffile.imsave("training_tiffs/gt/gt%i" % j,
                                np.asarray(self.conf_gt, dtype=np.float32)[j, :, :])
                tifffile.imsave("training_tiffs/grads_0/0grad%i"%j,
                               np.asarray(gradients,dtype=np.float32)[0,0,j,:,:,0])
                tifffile.imsave("training_tiffs/grads_1/0grad%i"%j,
                               np.asarray(gradients,dtype=np.float32)[0,0,j,:,:,1])
                tifffile.imsave("training_tiffs/grads_2/0grad%i"%j,
                               np.asarray(gradients,dtype=np.float32)[0,0,j,:,:,2])


    def save_model(self):
        print("Saving model...")
        self.model.save(self.save_loc + "model%i" % self.iteration)


    def __get_loss(self):


        if self.loss_schedule_flag==1:

            for i in range(0,np.shape(self.loss_schedule)[0]):

                if self.iteration<self.loss_schedule[i,0]:

                    return self.loss[i-1]

            return self.loss[-1]

        elif self.loss_schedule_flag==0:

            return self.loss


    def __get_critical_index(self,iterations):

        if self.loss_schedule_flag==1:

            for point in self.loss_schedule:

                if self.iteration<point[0]:
                    return point[0]

            return iterations

        else:
            return iterations


    def __train_iteration(self):

        std=0

        while std<0.4:
            raw_in, loss_info_in = self.provider.random_provider_loss_info()

            std=np.std(np.asarray(loss_info_in[:,:,:,:,0:3],dtype=np.float32))




        #bzxyc
        # if self.iteration%5==0:
        #     print(std)
        #     tifffile.imsave("training_tiffs/misc/raw",np.asarray(raw_in[0,8,:,:,0],dtype=np.float32))
        #     tifffile.imsave("training_tiffs/misc/aff0",np.asarray(loss_info_in[0,8,:,:,0],dtype=np.float32))
        #     tifffile.imsave("training_tiffs/misc/aff1",np.asarray(loss_info_in[0,8,:,:,1],dtype=np.float32))
        #     tifffile.imsave("training_tiffs/misc/aff2",np.asarray(loss_info_in[0,8,:,:,2],dtype=np.float32))
        #     tifffile.imsave("training_tiffs/misc/gt",np.asarray(loss_info_in[0,8,:,:,3],dtype=np.float32))

        self.model.fit(raw_in, loss_info_in, epochs=1)


    def reset_process(self):
        self.iteration=0
        self.model.reset()


    def calc_validation_loss(self):

        time_s=time.time()

        print("\nCalculating validation loss...")

        if len(self.validation_loss_list)==0:

            #self.model.compile(loss=self.__get_loss(), optimizer=self.optimizer, metrics=['accuracy'])
            self.__compile_model(loss=self.__get_loss())


            #blocks will go in zxyc
            provider_aff_0=assembler.assembler(self.validation_aff[0],0,self.input_shape[1:-1],None,1)
            provider_aff_1=assembler.assembler(self.validation_aff[1],0,self.input_shape[1:-1],None,1)
            provider_aff_2=assembler.assembler(self.validation_aff[2],0,self.input_shape[1:-1],None,1)

            provider_gt=assembler.assembler(self.validation_gt,0,self.input_shape[1:-1],None,1)

            provider_raw=assembler.assembler(self.validation_raw,0,self.input_shape[1:-1],None,1)


            aff_blocks_0=provider_aff_0.provide_all_raw_blocks()
            aff_blocks_1=provider_aff_1.provide_all_raw_blocks()
            aff_blocks_2=provider_aff_2.provide_all_raw_blocks()

            gt_blocks=provider_gt.provide_all_raw_blocks()

            raw_blocks=provider_raw.provide_all_raw_blocks()

            self.validation_data = [raw_blocks, gt_blocks, aff_blocks_0, aff_blocks_1, aff_blocks_2]

       #loss goes in bzxyc
        else:
            raw_blocks=self.validation_data[0]
            gt_blocks=self.validation_data[1]
            aff_blocks_0=self.validation_data[2]
            aff_blocks_1=self.validation_data[3]
            aff_blocks_2=self.validation_data[4]


        loss_info=np.zeros((1,4,self.input_shape[1],self.input_shape[2],self.input_shape[3]))
        raw=np.zeros((1,1,self.input_shape[1],self.input_shape[2],self.input_shape[3]))

        loss=0




        for i in range(0,np.shape(raw_blocks)[0]):
            aff_block_0=aff_blocks_0[i]
            aff_block_1=aff_blocks_1[i]
            aff_block_2=aff_blocks_2[i]

            gt_block=gt_blocks[i]

            raw_block=raw_blocks[i]

            loss_info[0,0]=aff_block_0
            loss_info[0,1]=aff_block_1
            loss_info[0,2]=aff_block_2

            loss_info[0,3]=gt_block

            raw[0,0]=raw_block
            info = self.model.evaluate(np.einsum("bczxy->bzxyc", raw), np.einsum("bczxy->bzxyc", loss_info), verbose=0)
            loss += info[0]

        #print("A")

        #info = self.model.evaluate(np.einsum("bczxy->bzxyc", raw), np.einsum("bczxy->bzxyc", loss_info), verbose=0)

        loss += info[0]

        print("\nTIME %f"%(time.time()-time_s))


        print("\nValidation loss: "+str(loss))

        return loss


    def decrement_lr(self):
        self.learning_rate=.1*self.learning_rate


    def reset_validation_loss_data(self):
        np.save("logs/loss_data%i"%self.iteration,self.validation_loss_list)
        self.validation_loss_list=[]




def get_layer_output_grad(model, inputs, outputs, layer=-1):
    """ Gets gradient a layer output for given inputs and outputs"""
    grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = k.backend.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad



