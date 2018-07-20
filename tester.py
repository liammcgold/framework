
import tensorflow as tf
import process
import numpy as np

new_weights=np.zeros((3,2))
new_weights[0] = [4.77092571, 0.55853532]
new_weights[1] = [25.34738088, 0.51006142]
new_weights[2] = [27.62890733, 0.50921526]

weights=np.zeros(3)
weights[0] = new_weights[0, 1] / new_weights[0, 0]
weights[1] = new_weights[1, 1] / new_weights[1, 0]
weights[2] = new_weights[2, 1] / new_weights[2, 0]

raw=np.load("data/spir_raw.npy")
gt=np.load("data/spir_gt.npy")
aff=np.load("data/spir_aff.npy")


def weighted_cross(target, logits):
    one_two =tf.add(tf.nn.weighted_cross_entropy_with_logits(targets=target[: ,: ,: ,: ,0] ,logits=logits[: ,: ,: ,: ,0]
                                                 ,pos_weight=weights[0]),
                   tf.nn.weighted_cross_entropy_with_logits(targets=target[: ,: ,: ,: ,1] ,logits=logits[: ,: ,: ,: ,1]
                                                 ,pos_weight=weights[1]))


    one_two_three =tf.add(one_two ,tf.nn.weighted_cross_entropy_with_logits(targets=target[: ,: ,: ,: ,2]
                                                                             ,logits=logits[: ,: ,: ,: ,2]
                                                                             ,pos_weight=weights[2]))

    return one_two_three





def train_from_scratch(saving_loc,loss=None,initial_lr=0.0025,gpus=1):



    n = 544

    proc = process.process(loss,
                           raw,
                           gt,
                           aff,
                           model_type="heavy paralell UNET",
                           precision="half",
                           save_loc=saving_loc,
                           saving_sched=[[0, 100], [1000, 10000], [100000, 20000]],
                           image_loc="tiffs/",
                           check_interval=100,
                           conf_coordinates=[[200, 216], [200, 328], [200, 328]],
                           learning_rate=initial_lr,
                           validation_frac=.2,
                           validation_interval=200,
                           gpus=gpus
                           )

    flag = proc.train(500000)
    while (not flag):
        proc.iteration = 0
        proc.learning_rate = proc.learning_rate * .1
        print("\n\tNEW LR = %f"%proc.learning_rate)
        flag = proc.train(500000)
