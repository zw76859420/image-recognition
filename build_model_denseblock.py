#-*- coding:utf-8 -*-
#author:zhangwei

import keras
from keras.models import Model
from keras.layers import Conv2D , MaxPooling2D , Input , Dense
from keras.layers import Activation , BatchNormalization , AveragePooling2D , concatenate


def dense_block(input , channels , k=4):
    bn1 = BatchNormalization(epsilon=1e-4)(input)
    relu = Activation(activation='relu')(bn1)
    conv1 = Conv2D(filters=4 * channels , kernel_size=[1 , 1] , padding='same' , kernel_initializer='he_normal' , use_bias=True)(relu)
    bn2 = BatchNormalization(epsilon=1e-4)(conv1)
    relu2 = Activation(activation='relu')(bn2)
    conv2 = Conv2D(filters=channels * k , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True)(relu2)
    return conv2

def transition_layer(input , channels):
    conv = Conv2D(filters=channels * 4 , kernel_size=[1 , 1] , padding='same' , kernel_initializer='he_normal' , use_bias=True)(input)
    pool = MaxPooling2D(pool_size=[2 , 2] , strides=[2 , 2])(conv)
    return pool

input_data = Input(shape=[1600 , 200 , 1] , name='input_data')
#conv1 = Conv2D(filters=32 , kernel_size=[5 , 1] , strides=[5 , 1], padding='valid' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(input_data)
x0 = Conv2D(filters=32 , kernel_size=[1 , 1] , padding='same' , use_bias=True , kernel_initializer='he_normal')(input_data)
x0 = MaxPooling2D(pool_size=[2 , 2])(x0)
b1_1 = dense_block(input=x0 , channels=8)
b1_1_conc = concatenate([x0 , b1_1] , axis=3)
b1_2 = dense_block(b1_1_conc , channels=8)
b1_2_conc = concatenate([x0 , b1_1 , b1_2] , axis=3)
b1_3 = dense_block(input=b1_2_conc , channels=8)
b1_3_conc = concatenate([b1_1 , b1_2 , b1_3],axis=3)
b1_4 = dense_block(b1_3_conc , 8)
b1_4_conc = concatenate([b1_2 , b1_3 , b1_4] , axis=3)
b1_5 = dense_block(b1_4_conc , 8)
b1_5_conc = concatenate([b1_3 , b1_4 , b1_5] , axis=3)
b1_6 = dense_block(b1_5_conc , 8)
x1 = transition_layer(b1_6 , 8)

b2_1 = dense_block(input=x1 , channels=16)
b2_1_conc = concatenate([x1 , b2_1] , axis=3)
b2_2 = dense_block(b2_1_conc , channels=16)
b2_2_conc = concatenate([x1 , b2_1 , b2_2] , axis=3)
b2_3 = dense_block(input=b2_2_conc , channels=16)
b2_3_conc = concatenate([b2_1 , b2_2 , b2_3],axis=3)
b2_4 = dense_block(b2_3_conc , 16)
b2_4_conc = concatenate([b2_2 , b2_3 , b2_4] , axis=3)
b2_5 = dense_block(b2_4_conc , 16)
b2_5_conc = concatenate([b2_3 , b2_4 , b2_5] , axis=3)
b2_6 = dense_block(b2_5_conc , 16)
x2 = transition_layer(b2_6 , 16)


model = Model(inputs=input_data , outputs=x2)
model.summary()