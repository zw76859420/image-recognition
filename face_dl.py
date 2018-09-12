#-*- coding:utf-8 -*-
#author:zhangwei

import random
import numpy as np
from sklearn.cross_validation import train_test_split
import keras as kr
from keras import backend as K
from keras.models import Model
from keras.layers import Dense , Conv2D , MaxPooling2D , Input , Reshape
from keras.layers import BatchNormalization , Dropout , regularizers , Flatten , Activation
from keras.optimizers import Adam , Adadelta , RMSprop , SGD
from keras.utils import np_utils
from keras.models import load_model

from load_dataset import load_dataset , resize_image , read_path



class Dataset():
    def __init__(self , pathname):

        self.train_images = None
        self.train_labels = None

        self.valid_images = None
        self.valid_labels = None

        self.test_images = None
        self.test_labels = None

        self.pathname = pathname
        self.input_shape = None

        pass

    def load(self , img_rows=64 , img_cols=64 , img_channels=3 , nb_classes=2):
        images , labels = load_dataset(self.pathname)
        train_images , valid_images , train_labels , valid_labels = train_test_split(images , labels , test_size=0.2 , random_state=random.randint(0 , 100))

        train_images = train_images.reshape(train_images.shape[0] , img_rows , img_cols , img_channels)
        valid_images = valid_images.reshape(valid_images.shape[0] , img_rows , img_cols , img_channels)

        self.input_shape = (img_rows , img_cols , img_channels)

        train_labels = np_utils.to_categorical(train_labels , num_classes=nb_classes)
        valid_labels = np_utils.to_categorical(valid_labels , num_classes=nb_classes)

        train_images.astype('float32')
        valid_images.astype('float32')

        train_images = train_images / 255
        valid_images = valid_images / 255

        self.train_images = train_images
        self.valid_images = valid_images
        self.train_labels = train_labels
        self.valid_labels = valid_labels

class ModelFace():
    def __init__(self):
        self.nb_calsses = 2
        self.filepath = '/home/zhangwei/'
        self.model = self.build_model()

    def build_model(self):

        input_data = Input(shape=[64 , 64 , 3])

        conv1 = Conv2D(filters=32 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(input_data)
        conv2 = Conv2D(filters=32 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(conv1)
        pool1 = MaxPooling2D(pool_size=[2 ,2] , strides=[2 , 2])(conv2)
        pool1 = Dropout(0.1)(pool1)

        conv3 = Conv2D(filters=64 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(pool1)
        conv4 = Conv2D(filters=64 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(conv3)
        pool2 = MaxPooling2D(pool_size=[2 , 2] ,strides=[2 , 2])(conv4)
        pool2 = Dropout(0.1)(pool2)

        flatten = Flatten()(pool2)
        flatten = Dropout(0.4)(flatten)
        dense1 = Dense(units=512 , activation='relu' , use_bias=True , kernel_initializer='he_normal')(flatten)
        dense1 = Dropout(0.1)(dense1)
        dense2 = Dense(units=self.nb_calsses , use_bias=True , kernel_initializer='he_normal')(dense1)
        pred = Activation(activation='softmax')(dense2)

        model_data = Model(inputs=input_data , outputs=pred)
        # model_data.summary()
        return model_data

    def train(self , dataset , batch_size=16 , nb_epoch=100 , data_augmentation=False):
        sgd = SGD(lr=0.01 , decay=1e-6 , momentum=0.9 , nesterov=True)
        self.model.compile(optimizer=sgd , loss='categorical_crossentropy' , metrics=['accuracy'])
        self.model.fit(dataset.train_images , dataset.train_labels , batch_size=batch_size , epochs=nb_epoch , validation_split=0.1 , verbose=1)
        self.save_model()

    def save_model(self , filepath='/home/zhangwei/face/myface.model.h5'):
        self.model.save(filepath=filepath)

    def load_mdoel(self , filepath='/home/zhangwei/face/myface.model.h5'):
        self.model = load_model(filepath=filepath)

    def Evaluate(self , dataset):
        score = self.model.evaluate(dataset.valid_images , dataset.valid_labels , verbose=1)
        print("%s:%.2f%%" % (self.model.metrics_names[1] , score[1] * 100))
        pass

    def face_predict_01(self , image):
        image = resize_image(image)
        image = image.astype('float32')
        image = image / 255
        result = self.model.predict(image)
        return result

if __name__ == "__main__":
    pathname='/home/zhangwei/data/'
    dataset = Dataset(pathname=pathname)
    dataset.load()

    model = ModelFace()
    # model.train(dataset=dataset)
    # model.Evaluate(dataset)
    model.load_mdoel()
    model.Evaluate(dataset)
