#-*- coding:utf-8 -*-
#author:zhangwei

import keras as kr
from keras.models import Model
from keras.layers import Input , Dense , multiply
# from keras.layers import merge , Multiply
from keras import backend as K

input_dims = 30
inputs = Input(shape=[input_dims ,])
attention_prob = Dense(units=input_dims , activation='softmax' , name='attention_vec')(inputs)
# attention_mul = merge([inputs , attention_prob])
attention_mul = multiply([inputs , attention_prob])
# print(attention_mul.shape)
attention_mul = Dense(64)(attention_mul)
# attention_mul = Dense(units=64)(inputs)
output = Dense(units=1 , activation='sigmoid')(attention_mul)
model = Model(inputs=inputs , outputs=output)

model.summary()


'''
"""
  首先我们先定义我们的输入维度是30；
  然后定义attention机制，attention机制实际上就是对于某一层输出的概率表示，用softmax激活函数进行计算每个向量之后的概率；
  最后我们使用merge层将inouts与attention定义的概率进行相乘，输出的维度一般要是跟输入的维度保持一致，也可以不一致，具体得看程序的要求；
  但是我有一个问题就是脚本变量中attention_mul第一个应该跟哪个层进行相连
  （attention_mul = merge([inputs , attention_prob] , output_shape=30 , mode='mul')）；
"""

from keras.layers import Dense, multiply, Input, Dropout
from keras.models import Model
import keras.backend as K
import numpy as np
from matplotlib import pyplot as plt

input_dim = 32
drop_rate = 0.5


def get_model():
    input_ = Input(shape=(input_dim,))
    ##attention begins 
    attention_probs = Dense(input_dim, name="attention_probs", activation="softmax")(input_)
    x = multiply([input_, attention_probs])
    ##attention ends
    x = Dropout(drop_rate)(x)

    x = Dense(16)(x)

    x = Dense(1, activation="sigmoid")(x)

    model = Model(input_, x)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
    return model


def get_data(n, attention_col):
    x = np.random.randn(n, input_dim)
    y = np.random.randint(0, 2, size=n)
    for col in attention_col:
        x[:, col] = y / float(len(attention_col))
    return x, y


def get_activation(model, layer_name, inputs):
    layer = [l for l in model.layers if l.name == layer_name][0]

    func = K.function([model.input], [layer.output])

    return func([inputs])


if __name__ == "__main__":
    x, y = get_data(100000, [1, 10, 20, 29])

    m = get_model()

    m.fit(x, y, batch_size=100, epochs=20, validation_split=0.4)

    test_x = x[0, :].reshape(1, input_dim)
    attentions = get_activation(m, "attention_probs", test_x)[0].flatten()

    plt.bar(np.arange(input_dim), attentions)
    plt.title("attention vetors ")
    plt.show()
'''

