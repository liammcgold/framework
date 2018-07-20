import tensorflow as tf
import numpy as np
import keras.backend as K
weight=0
import malis_loss
import tensorlayer


class loss(object):

    def __init__(self,shape=(16,128,128)):
        self.weights=[[1,1],[1,1],[1,1]]
        self.malis_object=malis_loss.mal(shape,self.weights)

    def set_weight(self,new_weights=[0]):
        #takes input as positive # vs negative # for each three affs, ex:

        new_weights=np.asarray(new_weights)

        if np.equal(new_weights,[0]).any():
            new_weights=np.zeros((3,2))
            new_weights[0] = [4.77092571, 0.55853532]
            new_weights[1] = [25.34738088, 0.51006142]
            new_weights[2] = [27.62890733, 0.50921526]


        self.weights[0]=new_weights[0,1]/new_weights[0,0]
        self.weights[1]=new_weights[1,1]/new_weights[1,0]
        self.weights[2]=new_weights[2,1]/new_weights[2,0]
        self.malis_object.weights=self.weights

    def print_weight(self):
        print(self.weights)

    def weighted_cross(self,target, logits):
        one_two=tf.add(tf.nn.weighted_cross_entropy_with_logits(targets=target[:,:,:,:,0],logits=logits[:,:,:,:,0],pos_weight=self.weights[0]),
                       tf.nn.weighted_cross_entropy_with_logits(targets=target[:,:,:,:,1],logits=logits[:,:,:,:,1],pos_weight=self.weights[1]))


        one_two_three=tf.add(one_two,tf.nn.weighted_cross_entropy_with_logits(targets=target[:,:,:,:,2],logits=logits[:,:,:,:,2],pos_weight=self.weights[2]))

        return one_two_three

    def dice(self, target, logits):

        return tensorlayer.cost.dice_coe(logits,target[:,:,:,:,0:3])






    def malis(self,target,logits):

        return self.malis_object.malis(target,logits)