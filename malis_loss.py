import malis
import keras
import tensorflow as tf
import numpy as np
import custom_loss as cl

nhood=malis.mknhood3d()
gt=np.load("data/spir_gt.npy")


class mal:
    def __init__(self,shape,weights,schedule=None,counter=0):
        self.nhood=malis.mknhood3d()
        self.shape=shape
        #self.weights=weights
        #self.cross=cl.loss()
        #self.cross.set_weight(weights)
        #self.schedule=schedule
        #self.counter=counter


    def malis(self, target,logits):

        #bzxyc

        aff=tf.slice(target,[0,0,0,0,0],[-1,-1,-1,-1,3])
        aff=tf.reshape(aff,(16,128,128,3))
        aff = tf.transpose(aff, [3, 0, 1, 2])

        gt=tf.slice(target,[0,0,0,0,3],[-1,-1,-1,-1,-1])
        gt=tf.reshape(gt,[16,128,128])

        out=tf.reshape(logits[0],[16,128,128,3])
        out=tf.transpose(out,[3,0,1,2])

        mal=malis.malis_loss_op(out,aff,gt,self.nhood)

        return mal

    def malis_cross(self,target,logits):
        # bzxyc

        aff = tf.slice(target, [0, 0, 0, 0, 0], [-1, -1, -1, -1, 3])
        aff = tf.reshape(aff, (16, 128, 128, 3))
        aff = tf.transpose(aff, [3, 0, 1, 2])

        gt = tf.slice(target, [0, 0, 0, 0, 3], [-1, -1, -1, -1, -1])
        gt = tf.reshape(gt, [16, 128, 128])

        out = tf.reshape(logits[0], [16, 128, 128, 3])
        out = tf.transpose(out, [3, 0, 1, 2])

        mal = malis.malis_loss_op(out, aff, gt, self.nhood)

        cross=self.cross.weighted_cross(tf.slice(target, [0, 0, 0, 0, 0], [-1, -1, -1, -1, 3]),logits)


        out=tf.add(mal,cross)

        return out

    def scheduled_loss(self):





        #schdeule format:
        '''
        dictionary with form 



        '''


