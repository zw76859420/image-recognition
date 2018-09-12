#-*- coding:utf-8 -*-
#author:zhangwei

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Bidirectional , LSTM , Dropout , Input , Dense , Activation , GRU , Reshape , Permute , Flatten
from keras.models import Model
from keras.optimizers import Adam

time_step = 28
input_dim = 28
lstm_units = 64

(x_train , y_train) , (x_test , y_test) = mnist.load_data('mnist.npz')
# print(y_train.shape)
x_train = x_train.reshape(-1 , 28 , 28) / 255
x_test = x_test.reshape(-1 , 28 , 28) / 255
y_train = np_utils.to_categorical(y=y_train , num_classes=10)
y_test = np_utils.to_categorical(y=y_test , num_classes=10)
# print(y_train.shape)

input = Input(shape=[time_step , input_dim])
input_drop = Dropout(0.1)(input)
lstm_out = Bidirectional(LSTM(units=lstm_units , return_sequences=True , kernel_initializer='he_normal') , name='Bilstm')(input_drop)
print(lstm_out.shape)
lstm_out = Flatten()(lstm_out)
lstm_drop = Dropout(0.1)(lstm_out)

output = Dense(units=10 , activation='softmax')(lstm_drop)

print(lstm_out.shape)

# model = Model(inputs=input , outputs=output)
# model.summary()


# adam = Adam(lr=0.01 , beta_1=0.9 , beta_2=0.999)
# model.compile(optimizer=adam , loss='categorical_crossentropy' , metrics=['accuracy'])
#
# model.summary()
#
# print("===========================Training============================")
#
# model.fit(x_train , y_train , batch_size=32 , epochs=100 , validation_split=0.2)
#
# print("===========================Testing===============================")
# loss , accuracy = model.evaluate(x=x_test , y=y_test)
# print("Loss" , loss)
# print("Accuracy" , accuracy)

# print(y_test.shape)