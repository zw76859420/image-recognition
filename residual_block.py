#-*- coding:utf-8 -*-
#author:zhangwei

import numpy as np
from keras.layers import Input ,  Activation , Add , Dense , ZeroPadding2D
from keras.layers import BatchNormalization , Flatten , Conv2D , MaxPooling2D , AveragePooling2D
from keras.models import Model
import keras.backend as K


def conv_block(x , f , filters , stage , block , s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1 , F2 , F3 = filters
    x_shortcut = x
    layer1 = Conv2D(filters=F1 , kernel_size=[1 , 1] , strides=[1 , 1] , padding='valid' , name=conv_name_base + '2a' , kernel_initializer='he_normal')(x_shortcut)
    layer1 = BatchNormalization(epsilon=1e-6 , name=bn_name_base + '2a')(layer1)
    layer1 = Activation(activation='relu')(layer1)

    layer2 = Conv2D(filters=F2 , kernel_size=[f , f] , strides=[1 , 1] , padding='same' , name=conv_name_base + '2b' , kernel_initializer='he_normal')(layer1)
    layer2 = BatchNormalization(epsilon=1e-6 , name=bn_name_base + '2b')(layer2)
    layer2 = Activation(activation='relu')(layer2)

    layer3 = Conv2D(filters=F3 , kernel_size=[1 , 1] , strides=[1 , 1] , padding='valid' , name=conv_name_base + '2c' , kernel_initializer='he_normal')(layer2)
    layer3 = BatchNormalization(epsilon=1e-6 , name=bn_name_base + '2c')(layer3)

    x_shortcut = Conv2D(filters=F3 , kernel_size=[1 ,1] , strides=[1 , 1] , padding='valid' , name=conv_name_base + '1' , kernel_initializer='he_normal')(x_shortcut)
    x_shortcut = BatchNormalization(epsilon=1e-6 , name=bn_name_base + '1')(x_shortcut)

    merge = Add()([layer3 , x_shortcut])
    merge = Activation(activation='relu')(merge)
    return merge

def identity_block(x , f , filters , stage , block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1 , F2 , F3 = filters
    x_shortcut = x

    layer1 = Conv2D(filters=F1 , kernel_size=[1 , 1] , strides=[1 , 1] , padding='valid' , name=conv_name_base + '2a' , kernel_initializer='he_normal')(x_shortcut)
    layer1 = BatchNormalization(epsilon=1e-6 , name=bn_name_base + '2a')(layer1)
    layer1 = Activation(activation='relu')(layer1)

    layer2 = Conv2D(filters=F2 , kernel_size=[f , f] , strides=[1 , 1] , padding='same' , name=conv_name_base + '2b' , kernel_initializer='he_normal')(layer1)
    layer2 = BatchNormalization(epsilon=1e-6 , name=bn_name_base + '2b')(layer2)
    layer2 = Activation(activation='relu')(layer2)

    layer3 = Conv2D(filters=F3 , kernel_size=[1 , 1] , strides=[1 , 1] , padding='valid' , name=conv_name_base + '2c' , kernel_initializer='he_normal')(layer2)
    layer3 = BatchNormalization(epsilon=1e-6 , name=bn_name_base + '2c')(layer3)

    merge = Add()([layer3 , x_shortcut])
    merge = Activation(activation='relu')(merge)

    return merge


x_input = Input(shape=[64 , 64 ,3] , name='Input')
x_pad = ZeroPadding2D(padding=[3 , 3])(x_input)

conv1 = Conv2D(filters=32 , kernel_size=[7 , 7] , strides=[2 , 2] , name='conv1' , kernel_initializer='he_normal')(x_pad)
conv1 = BatchNormalization(epsilon=1e-3 , name='bn_conv1')(conv1)
conv1 = Activation(activation='relu')(conv1)
pool1 = MaxPooling2D(pool_size=[3 , 3] , strides=[2 , 2])(conv1)


'''
   stage2搭建了9层卷积神经网络，先通过convblock，再通过两个identityblock
'''
conv2_block1 = conv_block(x=pool1 , f=3 , filters=[64 , 64 , 256] , stage=2 , block='a' , s=1)
conv2_block2 = identity_block(x=conv2_block1 , f=3 , filters=[64 , 64 , 256] , stage=2 , block='b')
conv2_block3 = identity_block(x=conv2_block2 , f=3 , filters=[64 , 64 , 256] , stage=2 , block='c')


'''
   stage3搭建了12层卷积神经网络，先通过convblock，再通过三个identityblock
'''
conv3_block1 = conv_block(x=conv2_block3 , f=3 , filters=[128 , 128 , 512] , stage=3 , block='a' , s=2)
conv3_block2 = identity_block(x=conv3_block1 , f=3 , filters=[128 , 128 , 512] , stage=3 , block='b')
conv3_block3 = identity_block(x=conv3_block2 , f=3 , filters=[128 , 128 , 512] , stage=3 , block='c')
conv3_block4 = identity_block(x=conv3_block3 , f=3 , filters=[128 , 128 , 512] , stage=3 , block='d')


model = Model(inputs=x_input , outputs=conv3_block4)
model.summary()


