import keras as k
import math
import numpy as np


def path_join(input_1,input_2,filters):

    s0=k.layers.add([input_1,input_2])

    r0=k.layers.LeakyReLU()(input_1)
    r1=k.layers.LeakyReLU()(s0)
    r2=k.layers.LeakyReLU()(input_2)

    s1=k.layers.add([r0,r1,r2])

    c0=k.layers.Conv3D(filters,(8,8,8),padding="same",activation="relu")(r0)
    c1=k.layers.Conv3D(filters,(8,8,8),padding="same",activation="relu")(s1)
    c2=k.layers.Conv3D(filters,(8,8,8),padding="same",activation="relu")(r2)

    s2=k.layers.add([c0,c1,c2])

    return s2


def large_kernel_make(verbose=0):

    raw_input=k.layers.Input((16,128,128,1))

    #########################
    #   Large Kernel Path   #
    #########################

    lks = 9

    # cl0
    cl0 = k.layers.Conv3D(8, (1, lks, lks), padding="same", activation="relu")(raw_input)
    cl01 = k.layers.Conv3D(8, (lks, lks, lks), padding="same", activation="relu")(cl0)
    cl0mp = k.layers.Conv3D(8, (lks, lks, lks), padding="same", activation="relu")(raw_input)
    cl0m = k.layers.add([cl0mp, cl01])
    cl0m = k.layers.BatchNormalization()(cl0m)

    # dl0
    dl0 = k.layers.MaxPool3D([1, 2, 2])(cl0m)

    # cl1
    cl1 = k.layers.Conv3D(8, (1, lks, lks), padding="same", activation="relu")(dl0)
    cl11 = k.layers.Conv3D(8, (lks, lks, lks), padding="same", activation="relu")(cl1)
    cl1mp = k.layers.Conv3D(8, (lks, lks, lks), padding="same", activation="relu")(dl0)
    cl1m = k.layers.add([cl1mp, cl11])
    cl1m = k.layers.BatchNormalization()(cl1m)

    # ml0
    ul0 = k.layers.UpSampling3D([1, 2, 2])(cl1m)
    ml0p = k.layers.LeakyReLU()(ul0)
    cl0mp0 = k.layers.LeakyReLU()(cl0m)
    ml0 = k.layers.add([ml0p, cl0mp0])

    # mcl0
    mcl0 = k.layers.Conv3D(3, (1, lks, lks), padding="same", activation="relu")(ml0)
    mcl01 = k.layers.Conv3D(3, (lks, lks, lks), padding="same", activation="relu")(mcl0)
    mcl0mp = k.layers.Conv3D(3, (lks, lks, lks), padding="same", activation="relu")(ml0)
    mcl0m = k.layers.add([mcl0mp, mcl01])
    mcl0m = k.layers.BatchNormalization()(mcl0m)

    o1 = k.layers.LeakyReLU()(mcl0m)

    #################
    #   Join Paths  #
    #################

    out=k.layers.LeakyReLU()(o1)
    out=k.layers.BatchNormalization()(out)

    model=k.models.Model(inputs=raw_input,outputs=out)

    if(verbose==1):
        print(model.summary())

    #k.utils.plot_model(model,"model_LK.png",show_shapes=True)

    return model


def merged_u_net_make(verbose=0):

    raw_input=k.layers.Input((16,128,128,1))


    #########################
    #   Small Kernel Path   #
    #########################

    sks=3

    #c0
    c0=k.layers.Conv3D(8,(1,sks,sks),padding="same",activation="relu")(raw_input)
    c00=k.layers.Conv3D(8,(sks,sks,sks),padding="same",activation="relu")(c0)
    c01=k.layers.Conv3D(8,(sks,sks,sks),padding="same",activation="relu")(c00)
    c0mp=k.layers.Conv3D(8,(sks,sks,sks),padding="same",activation="relu")(raw_input)
    c0m=k.layers.add([c0mp,c01])
    c0m=k.layers.BatchNormalization()(c0m)


    #d0
    d0=k.layers.MaxPool3D([1,2,2])(c0m)

    #c1
    c1 = k.layers.Conv3D(32, (1,sks,sks), padding="same", activation="relu")(d0)
    c10 = k.layers.Conv3D(32, (sks,sks,sks), padding="same", activation="relu")(c1)
    c11 = k.layers.Conv3D(32, (sks,sks,sks), padding="same", activation="relu")(c10)
    c1mp=k.layers.Conv3D(32, (sks,sks,sks), padding="same", activation="relu")(d0)
    c1m = k.layers.add([c1mp, c11])
    c1m=k.layers.BatchNormalization()(c1m)

    #d1
    d1=k.layers.MaxPool3D([1,2,2])(c1m)

    #c2
    c2 = k.layers.Conv3D(64, (1,sks,sks), padding="same", activation="relu")(d1)
    c20 = k.layers.Conv3D(64, (sks,sks,sks), padding="same", activation="relu")(c2)
    c21 = k.layers.Conv3D(64, (sks,sks,sks), padding="same", activation="relu")(c20)
    c2mp = k.layers.Conv3D(64, (sks,sks,sks), padding="same", activation="relu")(d1)
    c2m = k.layers.add([c2mp, c21])
    c2m=k.layers.BatchNormalization()(c2m)

    #d2
    d2=k.layers.MaxPool3D([1,2,2])(c2m)

    #c3
    c3 = k.layers.Conv3D(64, (1,sks,sks), padding="same", activation="relu")(d2)
    c30 = k.layers.Conv3D(64, (sks,sks,sks), padding="same", activation="relu")(c3)
    c31 = k.layers.Conv3D(64, (sks,sks,sks), padding="same", activation="relu")(c30)
    c3mp = k.layers.LeakyReLU()(d2)
    c3m = k.layers.add([c3mp, c31])
    c3m=k.layers.BatchNormalization()(c3m)


    #m0
    u0=k.layers.UpSampling3D([1,2,2])(c3m)
    m0p=k.layers.LeakyReLU()(u0)
    c2mp0=k.layers.LeakyReLU()(c2m)
    m0=k.layers.add([m0p,c2mp0])


    #mc0
    mc0 = k.layers.Conv3D(32, (1,sks,sks), padding="same", activation="relu")(m0)
    mc00 = k.layers.Conv3D(32, (sks,sks,sks), padding="same", activation="relu")(mc0)
    mc01 = k.layers.Conv3D(32, (sks,sks,sks), padding="same", activation="relu")(mc00)
    mc0mp = k.layers.Conv3D(32, (sks,sks,sks), padding="same", activation="relu")(m0)
    mc0m = k.layers.add([mc0mp, mc01])
    mc0m=k.layers.BatchNormalization()(mc0m)

    # m1
    u1 = k.layers.UpSampling3D([1, 2, 2])(mc0m)
    m1p = k.layers.LeakyReLU()(u1)
    c3mp0 = k.layers.LeakyReLU()(c1m)
    m1 = k.layers.add([m1p, c3mp0])

    # mc1
    mc1 = k.layers.Conv3D(8, (1,sks,sks), padding="same", activation="relu")(m1)
    mc10 = k.layers.Conv3D(8, (sks,sks,sks), padding="same", activation="relu")(mc1)
    mc11 = k.layers.Conv3D(8, (sks,sks,sks), padding="same", activation="relu")(mc10)
    mc1mp = k.layers.Conv3D(8, (sks,sks,sks), padding="same", activation="relu")(m1)
    mc1m = k.layers.add([mc1mp, mc11])
    mc1m=k.layers.BatchNormalization()(mc1m)

    # m2
    u2 = k.layers.UpSampling3D([1, 2, 2])(mc1m)
    m2p = k.layers.LeakyReLU()(u2)
    c0mp0 = k.layers.LeakyReLU()(c0m)
    m2 = k.layers.add([m2p, c0mp0])

    # mc2
    mc2 = k.layers.Conv3D(3, (1,sks,sks), padding="same", activation="relu")(m2)
    mc20 = k.layers.Conv3D(3, (sks,sks,sks), padding="same", activation="relu")(mc2)
    mc21 = k.layers.Conv3D(3, (sks,sks,sks), padding="same", activation="relu")(mc20)
    mc2mp = k.layers.Conv3D(3, (sks,sks,sks), padding="same", activation="relu")(m2)
    mc2m = k.layers.add([mc2mp, mc21])
    mc2m=k.layers.BatchNormalization()(mc2m)

    o0 = k.layers.LeakyReLU()(mc2m)


    #########################
    #   Large Kernel Path   #
    #########################

    lks = 9

    # cl0
    cl0 = k.layers.Conv3D(8, (1,lks,lks), padding="same", activation="relu")(raw_input)
    cl01 = k.layers.Conv3D(8, (lks,lks,lks), padding="same", activation="relu")(cl0)
    cl0mp = k.layers.Conv3D(8, (lks,lks,lks), padding="same", activation="relu")(raw_input)
    cl0m = k.layers.add([cl0mp, cl01])
    cl0m=k.layers.BatchNormalization()(cl0m)

    # dl0
    dl0 = k.layers.MaxPool3D([1, 2, 2])(cl0m)

    # cl1
    cl1 = k.layers.Conv3D(8, (1,lks,lks), padding="same", activation="relu")(dl0)
    cl11 = k.layers.Conv3D(8, (lks,lks,lks), padding="same", activation="relu")(cl1)
    cl1mp = k.layers.Conv3D(8, (lks,lks,lks), padding="same", activation="relu")(dl0)
    cl1m = k.layers.add([cl1mp, cl11])
    cl1m=k.layers.BatchNormalization()(cl1m)

    # ml0
    ul0 = k.layers.UpSampling3D([1, 2, 2])(cl1m)
    ml0p = k.layers.LeakyReLU()(ul0)
    cl0mp0 = k.layers.LeakyReLU()(cl0m)
    ml0 = k.layers.add([ml0p, cl0mp0])

    # mcl0
    mcl0 = k.layers.Conv3D(3, (1,lks,lks), padding="same", activation="relu")(ml0)
    mcl01 = k.layers.Conv3D(3, (lks,lks,lks), padding="same", activation="relu")(mcl0)
    mcl0mp = k.layers.Conv3D(3, (lks,lks,lks), padding="same", activation="relu")(ml0)
    mcl0m = k.layers.add([mcl0mp, mcl01])
    mcl0m= k.layers.BatchNormalization()(mcl0m)

    o1 = k.layers.LeakyReLU()(mcl0m)

    #################
    #   Join Paths  #
    #################

    join=path_join(o0,o1,3)
    out=k.layers.LeakyReLU()(join)
    out=k.layers.BatchNormalization(center=.5,)(out)
    out=k.layers.Conv3D(3,(lks,lks,lks),padding="same",activation="sigmoid")(out)

    model=k.models.Model(inputs=raw_input,outputs=out)

    if (verbose == 1):
        print(model.summary())

    #k.utils.plot_model(model, "model_MG.png", show_shapes=True)

    return model


def heavy_merged_u_net_make(verbose=0):

    raw_input=k.layers.Input((16,128,128,1))


    #########################
    #   Small Kernel Path   #
    #########################

    sks=3

    #c0
    c0=k.layers.Conv3D(8,(1,sks,sks),padding="same")(raw_input)
    drop0s=k.layers.Dropout(rate=0.5)(c0)
    drop_rel_0s=k.layers.LeakyReLU(trainable=True)(drop0s)
    c00=k.layers.Conv3D(8,(sks,sks,sks),padding="same",activation="relu")(drop_rel_0s)
    c01=k.layers.Conv3D(8,(sks,sks,sks),padding="same",activation="relu")(c00)
    c0mp=k.layers.Conv3D(8,(sks,sks,sks),padding="same",activation="relu")(raw_input)
    c0m=k.layers.add([c0mp,c01])
    c0m=k.layers.BatchNormalization()(c0m)


    #d0
    d0=k.layers.MaxPool3D([1,2,2])(c0m)

    #c1
    c1 = k.layers.Conv3D(32, (1,sks,sks), padding="same", activation="relu")(d0)
    drop1s = k.layers.Dropout(rate=0.5)(c1)
    drop_rel_1s = k.layers.LeakyReLU()(drop1s)
    c10 = k.layers.Conv3D(32, (sks,sks,sks), padding="same", activation="relu")(drop_rel_1s)
    c11 = k.layers.Conv3D(32, (sks,sks,sks), padding="same", activation="relu")(c10)
    c1mp=k.layers.Conv3D(32, (sks,sks,sks), padding="same", activation="relu")(d0)
    c1m = k.layers.add([c1mp, c11])
    c1m=k.layers.BatchNormalization()(c1m)

    #d1
    d1=k.layers.MaxPool3D([1,2,2])(c1m)

    #c2
    c2 = k.layers.Conv3D(64, (1,sks,sks), padding="same", activation="relu")(d1)
    drop2s = k.layers.Dropout(rate=0.5)(c2)
    drop_rel_2s = k.layers.LeakyReLU()(drop2s)
    c20 = k.layers.Conv3D(64, (sks,sks,sks), padding="same", activation="relu")(drop_rel_2s)
    c21 = k.layers.Conv3D(64, (sks,sks,sks), padding="same", activation="relu")(c20)
    c2mp = k.layers.Conv3D(64, (sks,sks,sks), padding="same", activation="relu")(d1)
    c2m = k.layers.add([c2mp, c21])
    c2m=k.layers.BatchNormalization()(c2m)

    #d2
    d2=k.layers.MaxPool3D([1,2,2])(c2m)

    #c3
    c3 = k.layers.Conv3D(64, (1,sks,sks), padding="same", activation="relu")(d2)
    drop3s = k.layers.Dropout(rate=0.5)(c3)
    drop_rel_3s = k.layers.LeakyReLU()(drop3s)
    c30 = k.layers.Conv3D(64, (sks,sks,sks), padding="same", activation="relu")(drop_rel_3s)
    c31 = k.layers.Conv3D(64, (sks,sks,sks), padding="same", activation="relu")(c30)
    c3mp = k.layers.LeakyReLU()(d2)
    c3m = k.layers.add([c3mp, c31])
    c3m=k.layers.BatchNormalization()(c3m)

#ADDED FOR HEAVY
################
    # d3
    d3 = k.layers.MaxPool3D([1, 2, 2])(c3m)

    #c4
    c4 = k.layers.Conv3D(128, (1,sks,sks), padding="same", activation="relu")(d3)
    drop4s = k.layers.Dropout(rate=0.5)(c4)
    drop_rel_4s = k.layers.LeakyReLU()(drop4s)
    c40 = k.layers.Conv3D(128, (sks,sks,sks), padding="same", activation="relu")(drop_rel_4s)
    c41 = k.layers.Conv3D(128, (sks,sks,sks), padding="same", activation="relu")(c40)
    c4mp =  k.layers.Conv3D(128, (sks,sks,sks), padding="same", activation="relu")(d3)
    c4m = k.layers.add([c4mp, c41])
    c4m=k.layers.BatchNormalization()(c4m)

    # d4
    d4= k.layers.MaxPool3D([1,2,2])(c4m)

    # c4
    c5 = k.layers.Conv3D(128, (1, sks, sks), padding="same", activation="relu")(d4)
    drop5s = k.layers.Dropout(rate=0.5)(c5)
    drop_rel_5s = k.layers.LeakyReLU()(drop5s)
    c50 = k.layers.Conv3D(128, (sks, sks, sks), padding="same", activation="relu")(drop_rel_5s)
    c51 = k.layers.Conv3D(128, (sks, sks, sks), padding="same", activation="relu")(c50)
    c5mp = k.layers.LeakyReLU()(d4)
    c5m = k.layers.add([c5mp, c51])
    c5m = k.layers.BatchNormalization()(c5m)

    #m_2
    u_2= k.layers.Deconv3D(128,(1,sks,sks),strides=[1,2,2],padding="same")(c5m)
    m_2p=k.layers.LeakyReLU()(u_2)
    c4mp_2=k.layers.LeakyReLU()(c4m)
    m_2=k.layers.add([m_2p,c4mp_2])

    # mc_2
    mc_2 = k.layers.Conv3D(64, (1, sks, sks), padding="same", activation="relu")(m_2)
    dropm0s = k.layers.Dropout(rate=0.5)(mc_2)
    drop_rel_m0s = k.layers.LeakyReLU()(dropm0s)
    mc_20 = k.layers.Conv3D(64, (sks, sks, sks), padding="same", activation="relu")(drop_rel_m0s)
    mc_21 = k.layers.Conv3D(64, (sks, sks, sks), padding="same", activation="relu")(mc_20)
    mc_2mp = k.layers.Conv3D(64, (sks, sks, sks), padding="same", activation="relu")(m_2)
    mc_2m = k.layers.add([mc_2mp, mc_21])
    mc_2m = k.layers.BatchNormalization()(mc_2m)


    #m_1
    u_1 = k.layers.Deconv3D(64,(1,sks,sks),strides=[1,2,2],padding="same")(mc_2m)
    m_1p = k.layers.LeakyReLU()(u_1)
    c3mp_1 = k.layers.LeakyReLU()(c3m)
    m_1 = k.layers.add([m_1p, c3mp_1])

    # mc_1
    mc_1 = k.layers.Conv3D(64, (1, sks, sks), padding="same", activation="relu")(m_1)
    dropm1s = k.layers.Dropout(rate=0.5)(mc_1)
    drop_rel_m1s = k.layers.LeakyReLU()(dropm1s)
    mc_10 = k.layers.Conv3D(64, (sks, sks, sks), padding="same", activation="relu")(drop_rel_m1s)
    mc_11 = k.layers.Conv3D(64, (sks, sks, sks), padding="same", activation="relu")(mc_10)
    mc_1mp = k.layers.Conv3D(64, (sks, sks, sks), padding="same", activation="relu")(m_1)
    mc_1m = k.layers.add([mc_1mp, mc_11])
    mc_1m = k.layers.BatchNormalization()(mc_1m)




##############




    #m0
    u0= k.layers.Deconv3D(64,(1,sks,sks),strides=[1,2,2],padding="same")(mc_1m)
    m0p=k.layers.LeakyReLU()(u0)
    c2mp0=k.layers.LeakyReLU()(c2m)
    m0=k.layers.add([m0p,c2mp0])


    #mc0
    mc0 = k.layers.Conv3D(32, (1,sks,sks), padding="same", activation="relu")(m0)
    dropm2s = k.layers.Dropout(rate=0.5)(mc0)
    drop_rel_m2s = k.layers.LeakyReLU()(dropm2s)
    mc00 = k.layers.Conv3D(32, (sks,sks,sks), padding="same", activation="relu")(drop_rel_m2s)
    mc01 = k.layers.Conv3D(32, (sks,sks,sks), padding="same", activation="relu")(mc00)
    mc0mp = k.layers.Conv3D(32, (sks,sks,sks), padding="same", activation="relu")(m0)
    mc0m = k.layers.add([mc0mp, mc01])
    mc0m=k.layers.BatchNormalization()(mc0m)

    # m1
    u1 = k.layers.Deconv3D(32,(1,sks,sks),strides=[1,2,2],padding="same")(mc0m)
    m1p = k.layers.LeakyReLU()(u1)
    c3mp0 = k.layers.LeakyReLU()(c1m)
    m1 = k.layers.add([m1p, c3mp0])

    # mc1
    mc1 = k.layers.Conv3D(8, (1,sks,sks), padding="same", activation="relu")(m1)
    dropm3s = k.layers.Dropout(rate=0.5)(mc1)
    drop_rel_m3s = k.layers.LeakyReLU()(dropm3s)
    mc10 = k.layers.Conv3D(8, (sks,sks,sks), padding="same", activation="relu")(drop_rel_m3s)
    mc11 = k.layers.Conv3D(8, (sks,sks,sks), padding="same", activation="relu")(mc10)
    mc1mp = k.layers.Conv3D(8, (sks,sks,sks), padding="same", activation="relu")(m1)
    mc1m = k.layers.add([mc1mp, mc11])
    mc1m=k.layers.BatchNormalization()(mc1m)

    # m2
    u2 =  k.layers.Deconv3D(8,(1,sks,sks),strides=[1,2,2],padding="same")(mc1m)
    m2p = k.layers.LeakyReLU()(u2)
    c0mp0 = k.layers.LeakyReLU()(c0m)
    m2 = k.layers.add([m2p, c0mp0])

    # mc2
    mc2 = k.layers.Conv3D(3, (1,sks,sks), padding="same", activation="relu")(m2)
    dropm4s = k.layers.Dropout(rate=0.5)(mc2)
    drop_rel_m4s = k.layers.LeakyReLU()(dropm4s)
    mc20 = k.layers.Conv3D(3, (sks,sks,sks), padding="same", activation="relu")(drop_rel_m4s)
    mc21 = k.layers.Conv3D(3, (sks,sks,sks), padding="same", activation="relu")(mc20)
    mc2mp = k.layers.Conv3D(3, (sks,sks,sks), padding="same", activation="relu")(m2)
    mc2m = k.layers.add([mc2mp, mc21])
    mc2m=k.layers.BatchNormalization()(mc2m)

    o0 = k.layers.LeakyReLU()(mc2m)


    #########################
    #   Large Kernel Path   #
    #########################

    lks = 9

    # cl0
    cl0 = k.layers.Conv3D(8, (1,lks,lks), padding="same", activation="relu")(raw_input)
    drop0l = k.layers.Dropout(rate=0.5)(cl0)
    drop_rel_0l = k.layers.LeakyReLU()(drop0l)
    cl01 = k.layers.Conv3D(8, (lks,lks,lks), padding="same", activation="relu")(drop_rel_0l)
    cl0mp = k.layers.Conv3D(8, (lks,lks,lks), padding="same", activation="relu")(raw_input)
    cl0m = k.layers.add([cl0mp, cl01])
    cl0m=k.layers.BatchNormalization()(cl0m)

    # dl0
    dl0 = k.layers.MaxPool3D([1, 2, 2])(cl0m)

    # cl1
    cl1 = k.layers.Conv3D(8, (1,lks,lks), padding="same", activation="relu")(dl0)
    drop1l = k.layers.Dropout(rate=0.5)(cl1)
    drop_rel_1l = k.layers.LeakyReLU()(drop1l)
    cl11 = k.layers.Conv3D(8, (lks,lks,lks), padding="same", activation="relu")(drop_rel_1l)
    cl1mp = k.layers.Conv3D(8, (lks,lks,lks), padding="same", activation="relu")(dl0)
    cl1m = k.layers.add([cl1mp, cl11])
    cl1m=k.layers.BatchNormalization()(cl1m)


#ADDED FOR HEAVY
################

    #dl1
    dl1 = k.layers.MaxPool3D([1,2,2])(cl1m)

    #cl2
    cl2 = k.layers.Conv3D(8, (1, lks, lks), padding="same", activation="relu")(dl1)
    drop2l = k.layers.Dropout(rate=0.5)(cl2)
    drop_rel_2l = k.layers.LeakyReLU()(drop2l)
    cl21 = k.layers.Conv3D(8, (lks, lks, lks), padding="same", activation="relu")(drop_rel_2l)
    cl2mp = k.layers.Conv3D(8, (lks, lks, lks), padding="same", activation="relu")(dl1)
    cl2m = k.layers.add([cl2mp, cl21])
    cl2m = k.layers.BatchNormalization()(cl2m)

    # ml0
    ul =  k.layers.Deconv3D(8,(1,sks,sks),strides=[1,2,2],padding="same")(cl2m)
    mlp = k.layers.LeakyReLU()(ul)
    clmp0 = k.layers.LeakyReLU()(cl1m)
    ml = k.layers.add([mlp, clmp0])

    #mcl
    mcl = k.layers.Conv3D(8, (1, lks, lks), padding="same", activation="relu")(ml)
    dropm0l = k.layers.Dropout(rate=0.5)(mcl)
    drop_rel_m0l = k.layers.LeakyReLU()(dropm0l)
    mcl1 = k.layers.Conv3D(8, (lks, lks, lks), padding="same", activation="relu")(drop_rel_m0l)
    mclmp = k.layers.Conv3D(8, (lks, lks, lks), padding="same", activation="relu")(ml)
    mclm = k.layers.add([mclmp, mcl1])
    mclm = k.layers.BatchNormalization()(mclm)



#########
    # ml0
    ul0 =  k.layers.Deconv3D(8,(1,sks,sks),strides=[1,2,2],padding="same",trainable=True)(mclm)
    ml0p = k.layers.LeakyReLU()(ul0)
    cl0mp0 = k.layers.LeakyReLU()(cl0m)
    ml0 = k.layers.add([ml0p, cl0mp0])

    # mcl0
    mcl0 = k.layers.Conv3D(3, (1,lks,lks), padding="same", activation="relu")(ml0)
    dropm1l = k.layers.Dropout(rate=0.5)(mcl0)
    drop_rel_m1l = k.layers.LeakyReLU()(dropm1l)
    mcl01 = k.layers.Conv3D(3, (lks,lks,lks), padding="same", activation="relu")(drop_rel_m1l)
    mcl0mp = k.layers.Conv3D(3, (lks,lks,lks), padding="same", activation="relu")(ml0)
    mcl0m = k.layers.add([mcl0mp, mcl01])
    mcl0m= k.layers.BatchNormalization()(mcl0m)

    o1 = k.layers.LeakyReLU()(mcl0m)

    #################
    #   Join Paths  #
    #################

    join=path_join(o0,o1,3)
    out=k.layers.LeakyReLU()(join)
    out=k.layers.BatchNormalization(center=.5,)(out)
    out=k.layers.Conv3D(3,(lks,lks,lks),padding="same",activation="sigmoid")(out)

    model=k.models.Model(inputs=raw_input,outputs=out)

    if (verbose == 1):
        print(model.summary())

    #k.utils.plot_model(model, "model_heavy.png", show_shapes=True)

    return model


def small_kernel_u_net_make(verbose=0):

    raw_input=k.layers.Input((16,128,128,1))
    #########################
    #   Small Kernel Path   #
    #########################

    sks = 3

    # c0
    c0 = k.layers.Conv3D(8, (1, sks, sks), padding="same", activation="relu")(raw_input)
    c00 = k.layers.Conv3D(8, (sks, sks, sks), padding="same", activation="relu")(c0)
    c01 = k.layers.Conv3D(8, (sks, sks, sks), padding="same", activation="relu")(c00)
    c0mp = k.layers.Conv3D(8, (sks, sks, sks), padding="same", activation="relu")(raw_input)
    c0m = k.layers.add([c0mp, c01])
    c0m = k.layers.BatchNormalization()(c0m)

    # d0
    d0 = k.layers.MaxPool3D([1, 2, 2])(c0m)

    # c1
    c1 = k.layers.Conv3D(32, (1, sks, sks), padding="same", activation="relu")(d0)
    c10 = k.layers.Conv3D(32, (sks, sks, sks), padding="same", activation="relu")(c1)
    c11 = k.layers.Conv3D(32, (sks, sks, sks), padding="same", activation="relu")(c10)
    c1mp = k.layers.Conv3D(32, (sks, sks, sks), padding="same", activation="relu")(d0)
    c1m = k.layers.add([c1mp, c11])
    c1m = k.layers.BatchNormalization()(c1m)

    # d1
    d1 = k.layers.MaxPool3D([1, 2, 2])(c1m)

    # c2
    c2 = k.layers.Conv3D(64, (1, sks, sks), padding="same", activation="relu")(d1)
    c20 = k.layers.Conv3D(64, (sks, sks, sks), padding="same", activation="relu")(c2)
    c21 = k.layers.Conv3D(64, (sks, sks, sks), padding="same", activation="relu")(c20)
    c2mp = k.layers.Conv3D(64, (sks, sks, sks), padding="same", activation="relu")(d1)
    c2m = k.layers.add([c2mp, c21])
    c2m = k.layers.BatchNormalization()(c2m)

    # d2
    d2 = k.layers.MaxPool3D([1, 2, 2])(c2m)

    # c3
    c3 = k.layers.Conv3D(64, (1, sks, sks), padding="same", activation="relu")(d2)
    c30 = k.layers.Conv3D(64, (sks, sks, sks), padding="same", activation="relu")(c3)
    c31 = k.layers.Conv3D(64, (sks, sks, sks), padding="same", activation="relu")(c30)
    c3mp = k.layers.LeakyReLU()(d2)
    c3m = k.layers.add([c3mp, c31])
    c3m = k.layers.BatchNormalization()(c3m)

    # m0
    u0 = k.layers.UpSampling3D([1, 2, 2])(c3m)
    m0p = k.layers.LeakyReLU()(u0)
    c2mp0 = k.layers.LeakyReLU()(c2m)
    m0 = k.layers.add([m0p, c2mp0])

    # mc0
    mc0 = k.layers.Conv3D(32, (1, sks, sks), padding="same", activation="relu")(m0)
    mc00 = k.layers.Conv3D(32, (sks, sks, sks), padding="same", activation="relu")(mc0)
    mc01 = k.layers.Conv3D(32, (sks, sks, sks), padding="same", activation="relu")(mc00)
    mc0mp = k.layers.Conv3D(32, (sks, sks, sks), padding="same", activation="relu")(m0)
    mc0m = k.layers.add([mc0mp, mc01])
    mc0m = k.layers.BatchNormalization()(mc0m)

    # m1
    u1 = k.layers.UpSampling3D([1, 2, 2])(mc0m)
    m1p = k.layers.LeakyReLU()(u1)
    c3mp0 = k.layers.LeakyReLU()(c1m)
    m1 = k.layers.add([m1p, c3mp0])

    # mc1
    mc1 = k.layers.Conv3D(8, (1, sks, sks), padding="same", activation="relu")(m1)
    mc10 = k.layers.Conv3D(8, (sks, sks, sks), padding="same", activation="relu")(mc1)
    mc11 = k.layers.Conv3D(8, (sks, sks, sks), padding="same", activation="relu")(mc10)
    mc1mp = k.layers.Conv3D(8, (sks, sks, sks), padding="same", activation="relu")(m1)
    mc1m = k.layers.add([mc1mp, mc11])
    mc1m = k.layers.BatchNormalization()(mc1m)

    # m2
    u2 = k.layers.UpSampling3D([1, 2, 2])(mc1m)
    m2p = k.layers.LeakyReLU()(u2)
    c0mp0 = k.layers.LeakyReLU()(c0m)
    m2 = k.layers.add([m2p, c0mp0])

    # mc2
    mc2 = k.layers.Conv3D(3, (1, sks, sks), padding="same", activation="relu")(m2)
    mc20 = k.layers.Conv3D(3, (sks, sks, sks), padding="same", activation="relu")(mc2)
    mc21 = k.layers.Conv3D(3, (sks, sks, sks), padding="same", activation="relu")(mc20)
    mc2mp = k.layers.Conv3D(3, (sks, sks, sks), padding="same", activation="relu")(m2)
    mc2m = k.layers.add([mc2mp, mc21])
    mc2m = k.layers.BatchNormalization()(mc2m)

    out=k.layers.LeakyReLU()(mc2m)

    out=k.layers.BatchNormalization()(out)



    model=k.models.Model(inputs=raw_input,outputs=out)

    if (verbose == 1):
        print(model.summary())

    #k.utils.plot_model(model, "model_SK.png", show_shapes=True)

    return model


def autoencoder(input_shape,compression_factor,kernel_size=3,features=4):

    input_shape=np.asarray(input_shape)
    input_shape=input_shape.tolist()
    input_shape.append(1)
    print(input_shape)
    input=k.layers.Input(input_shape)


    def prime_fac(n):
        prime_factors = []
        i = 1
        while (i <= n):
            k = 0
            if (n % i == 0):
                j = 1
                while (j <= i):
                    if (i % j == 0):
                        k = k + 1
                    j = j + 1
                if (k == 2):
                    prime_factors.append(i)
            i = i + 1

        factors=[]

        for factor in prime_factors:
            a=n
            a = a / factor
            while a%1==0:
                factors.append(factor)
                a=a/factor

        return factors


    if np.shape(input_shape)[0]==3:

        def stable_layer(input):

            return k.layers.Conv2D(features,(kernel_size,kernel_size),padding="same",activation="relu")(input)

        def down_layer(input,factor):

            return k.layers.MaxPool2D([factor,factor])(input)

        def up_layer(input,factor):

            return k.layers.Deconv2D(features,(kernel_size,kernel_size),padding="same",activation="relu",strides=(factor,factor))(input)

    elif np.shape(input_shape)[0]==4:

        def stable_layer(input):

            return k.layers.Conv3D(features, (kernel_size, kernel_size,kernel_size), padding="same", activation="relu")(input)

        def down_layer(input, factor):

            return k.layers.MaxPool3D([factor,factor,factor])(input)

        def up_layer(input, factor):

            return k.layers.Deconv3D(features, (kernel_size, kernel_size,kernel_size), padding="same", activation="relu",strides=(factor, factor,factor))(input)

    prime_factors = prime_fac(compression_factor)

    layer=input

    print(layer.shape)
    #ENCODING
    for factor in prime_factors:

        for i in range(0,factor):

            layer=stable_layer(layer)

        layer=down_layer(layer,factor)

    latent_variable_layer=k.layers.LeakyReLU(name="LATENT_LAYER")(layer)

    layer=latent_variable_layer

    prime_factors=prime_factors[::-1]

    #DECODING
    for factor in prime_factors:

        for i in range(0, factor):
            layer = stable_layer(layer)

        layer = up_layer(layer, factor)

    out=stable_layer(layer)

    model=k.models.Model(input=input,output=out)

    k.utils.plot_model(model, "autoencoder.png", show_shapes=True)

    return model




