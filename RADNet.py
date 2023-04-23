# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 17:32:29 2023

@author: Administrator
"""
import keras
from keras.models import Model
from keras.layers import Input, Activation, Conv2D, Dropout
from keras.layers import MaxPooling2D, BatchNormalization,Lambda,multiply,SeparableConv2D,GlobalAveragePooling2D,MaxPool2D
from keras.layers import UpSampling2D,GlobalMaxPool2D,Dense,add,Reshape,AveragePooling2D,GlobalAvgPool2D,Add,Conv1D
from keras.layers import concatenate,Concatenate,multiply,subtract
from keras.optimizers import SGD, rmsprop, Adam, Adamax
from keras.layers import add
from keras.engine.topology import Layer
import math
import keras.backend as K
import tensorflow as tf
from loss import binary_focal_loss

def res_block(inputlayer,kernelnum,name=None):
    conv = Conv2D(kernelnum, 1, padding='same', kernel_initializer='he_normal')(inputlayer)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv1 = Conv2D(kernelnum, 3,padding='same', kernel_initializer='he_normal')(conv)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(kernelnum, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)  
    if name:        
        out = add([conv, conv2],name=name)
    else:
        out = add([conv, conv2])
    return out

def neg(layer):
    y = Lambda(lambda x: x*(-1))(layer)
   
    return y


def ram(high_layer,low_layer,channel_num):
    high_layer = UpSampling2D(size=(2, 2))(high_layer)  
    concat = concatenate([high_layer,low_layer])  
    concat = Conv2D(channel_num,3, padding = 'same', kernel_initializer = 'he_normal')(concat) 
    x_neg = neg(concat) 
    rever = Conv2D(channel_num,1,  padding = 'same', kernel_initializer = 'he_normal')(x_neg)
    rever = Conv2D(channel_num,1,  activation = 'relu',  padding = 'same', kernel_initializer = 'he_normal')(x_neg)
    rever = Conv2D(channel_num, 3, padding = 'same', kernel_initializer = 'he_normal')(rever) 
    rever1 = neg(rever) 
    concat = BatchNormalization()(concat)
    concat = Activation('relu')(concat)
    concat = Conv2D(channel_num,3,  padding = 'same', kernel_initializer = 'he_normal')(concat)
    obj_sig = Activation('sigmoid')(concat)
    rev_sig = Lambda(lambda x: 1-x)(obj_sig)
    x = multiply([rever1,rev_sig])
    x = add([x ,concat]) 
    return x

def BAM1(seg_layer,edge_layer,name):    
    x_s = Conv2D(32, 1, padding='same', kernel_initializer='he_normal')(seg_layer)
    x_s = BatchNormalization()(x_s)
    x_s = Activation('relu')(x_s) 
    x_s = Conv2D(1, 1, padding='same', kernel_initializer='he_normal')(x_s)
    x_s = Activation('sigmoid')(x_s)    
    x_e = multiply([edge_layer,x_s])   
    x = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x_e)
    x = BatchNormalization()(x)
    x = Activation('relu',name=name)(x) 
    return x

def bgm(s_layer,b_layer,channel,name= None):
    x =Lambda(lambda x: tf.ones_like(x)*0.5)(b_layer)
    b_layer1_1 = subtract([b_layer,x])
    b_layer1 = Activation('relu')(b_layer1_1)
    s_layer2 = multiply([s_layer,b_layer1])
    s_gap_layer = GlobalAveragePooling2D()(s_layer2)
    s_gap_layer = Activation('sigmoid')(s_gap_layer)
    s_layer3 = multiply([s_layer,s_gap_layer])
    s_layer4 = Conv2D(channel, 1, padding='same', kernel_initializer='he_normal')(s_layer3)
    s_layer5 = multiply([s_layer4,b_layer])
    s_layer6 = add([s_layer5,s_layer])
    out = Conv2D(channel, 3,activation='relu', padding='same', kernel_initializer='he_normal',name=name)(s_layer6)
    return out

def RADNet(classes, input_size, epochs, batch_size, LR,
          Falg_summary=False, Falg_plot_model=False, pretrained_weights = None):
    # ======================================编码器=============================================
    input1 = Input(input_size)
    conv = Conv2D(64, 7, padding='same', kernel_initializer='he_normal')(input1)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    encode1 = res_block(conv, 64,name='res1') 
    pool1 = MaxPooling2D(pool_size=(2, 2))(encode1)
    encode2 = res_block(pool1, 128,name='res2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(encode2)
    encode3 = res_block(pool2, 256,name='res3')
    pool3 = MaxPooling2D(pool_size=(2, 2))(encode3)
    encode4 = res_block(pool3, 512,name='res4')
    pool4 = MaxPooling2D(pool_size=(2, 2))(encode4)
    encode5 = res_block(pool4, 512,name='res5')
    # ======================================边缘流===============================================   
    edge1 = res_block(conv, 32,name='edge')
    up1 = UpSampling2D(size=(4, 4),name = 'encode_up1')(encode3)
    bam1 = BAM1(up1,edge1,name='sig1')
    
    edge2 = res_block(bam1, 32,name='edge1')
    up2 = UpSampling2D(size=(8, 8),name = 'encode_up2')(encode4) 
    bam2 = BAM1(up2,edge2,name='sig2')
    
    edge3 = res_block(bam2, 32,name='edge2')
    up3 = UpSampling2D(size=(16, 16),name = 'encode_up3')(encode5) 
    bam3 = BAM1(up3,edge3,name='sig3')   
    #edge4 = res_block(bam3, 32,name='edge3')
    edge_out = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='edge_conv_out')(bam3)
    edge_out = Conv2D(1, 1, activation='sigmoid', name='edge_out')(edge_out)
    # ======================================解码器=============================================
    conv11 = ram(encode5,encode4,512)  
    conv11 =  Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv4')(conv11)
  
    conv22 = ram(conv11,encode3,256)  
    conv22 =  Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv3')(conv22) 
    
    conv33 = ram(conv22,encode2,128)  
    conv33 =  Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv2')(conv33) 
    
    edge_out0 = MaxPooling2D(pool_size=(2, 2))(edge_out)
    conv = bgm(conv33,edge_out0,128)
 
    conv44 = ram(conv,encode1,64)  
    conv44 =  Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv1')(conv44) 

    conv = bgm(conv44,edge_out,64) 
    # ======================================输出=============================================
    out = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',name = 'conv_out')(conv)
    #=========================================融合模块====================================================================

    out = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(out) 
    out = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='out')(out)
    seg_out = Conv2D(1, 1, activation='sigmoid', name='seg_out')(out)  
    
    model = Model(inputs=input1, outputs=[seg_out,edge_out])
    opt = Adam(lr = 1e-4)
    model.compile(optimizer=opt,
                  loss={'seg_out': 'binary_crossentropy',
                        'edge_out': binary_focal_loss(alpha=.1, gamma=4),
                       
                        },
                  loss_weights={
                      'seg_out': 1,
                      'edge_out': 0.6,
                  },
                  metrics={'seg_out': ['accuracy'],
                           'edge_out': ['accuracy'],
                          })
    if Falg_summary:
        model.summary()
    if Falg_plot_model:
        keras.utils.plot_model(model, to_file='logs/DANet_model.png', show_shapes=True, show_layer_names=True)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model


RADNet(2, (128,128,3), epochs=1, batch_size=2, LR=1, Falg_summary=True, Falg_plot_model=False)