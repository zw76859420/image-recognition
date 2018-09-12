#-*- coding:utf-8 -*-
#author:zhangwei

import random
import numpy as np
from sklearn.cross_validation import train_test_split
import keras as kr
from keras import backend as K
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense , Conv2D , MaxPooling2D , Input , Reshape
from keras.layers import BatchNormalization , Dropout , regularizers , Flatten , Activation , GlobalAveragePooling2D
from keras.optimizers import Adam , Adadelta , RMSprop , SGD
from keras.utils import np_utils
from keras.models import load_model

from data_preprocess import *

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

    def load(self , img_rows=128 , img_cols=128 , img_channels=3 , nb_classes=2):
        images , labels = load_dataset(self.pathname)
        train_images , valid_images , train_labels , valid_labels = train_test_split(images , labels , test_size=0.3 , random_state=random.randint(0 , 100))
        valid_images , test_images , valid_labels , test_labels = train_test_split(valid_images , valid_labels , test_size=0.5 , random_state=random.randint(0 , 100))

        train_images = train_images.reshape(train_images.shape[0] , img_rows , img_cols , img_channels)
        valid_images = valid_images.reshape(valid_images.shape[0] , img_rows , img_cols , img_channels)
        test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
        # print(valid_images.shape)

        self.input_shape = (img_rows , img_cols , img_channels)

        train_labels = np_utils.to_categorical(train_labels , num_classes=nb_classes)
        valid_labels = np_utils.to_categorical(valid_labels , num_classes=nb_classes)
        test_labels = np_utils.to_categorical(test_labels , num_classes=nb_classes)
        # print(test_labels)

        train_images.astype('float32')
        valid_images.astype('float32')
        test_images.astype('float32')

        train_images = train_images / 255
        valid_images = valid_images / 255
        test_images = test_images / 255

        self.train_images = train_images
        self.valid_images = valid_images
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.test_images = test_images
        self.test_labels = test_labels

class ModelFace():
    def __init__(self):
        self.nb_calsses = 2
        # self.filepath = '/home/zhangwei/'
        self.model = self.build_model()

    def build_model(self):

        input_data = Input(shape=[128 , 128 , 3])

        conv1 = Conv2D(filters=32 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(input_data)
        conv1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=32 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(conv1)
        conv2 = BatchNormalization()(conv2)
        pool1 = MaxPooling2D(pool_size=[2 ,2] , strides=[2 , 2])(conv2)
        pool1 = Dropout(0.1)(pool1)

        conv3 = Conv2D(filters=64 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(pool1)
        conv3 = BatchNormalization()(conv3)
        conv4 = Conv2D(filters=64 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(conv3)
        conv4 = BatchNormalization()(conv4)
        pool2 = MaxPooling2D(pool_size=[2 , 2] ,strides=[2 , 2])(conv4)
        pool2 = Dropout(0.1)(pool2)

        conv5 = Conv2D(filters=128 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(pool2)
        conv5 = BatchNormalization()(conv5)
        conv6 = Conv2D(filters=128 , kernel_size=[3 , 3] , padding='same' , kernel_initializer='he_normal' , use_bias=True , activation='relu')(conv5)
        conv6 = BatchNormalization()(conv6)
        pool3 = GlobalAveragePooling2D()(conv6)

        dense1 = Dense(units=128 , activation='relu' , use_bias=True , kernel_initializer='he_normal')(pool3)
        dense1 = Dropout(0.1)(dense1)
        dense2 = Dense(units=256 , activation='relu' , use_bias=True , kernel_initializer='he_normal')(dense1)
        dense2 = Dropout(rate=0.2)(dense2)
        dense3 = Dense(units=self.nb_calsses , use_bias=True , kernel_initializer='he_normal')(dense2)
        pred = Activation(activation='softmax')(dense3)

        model_data = Model(inputs=input_data , outputs=pred)
        # model_data.summary()
        return model_data

    def train(self , dataset , batch_size=32 , nb_epoch=1000 , data_augmentation=False):
        sgd = SGD(lr=0.01 , decay=1e-6 , momentum=0.9 , nesterov=True)
        adam = Adam(lr=0.001)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy' , metrics=['accuracy'])
        if not data_augmentation:
            self.model.fit(dataset.train_images , dataset.train_labels , batch_size=batch_size , epochs=nb_epoch , validation_split=0.1 , verbose=1)
        else:
            datagen = ImageDataGenerator(
                featurewise_center=False ,
                samplewise_center=False ,
                featurewise_std_normalization=False ,
                samplewise_std_normalization=False ,
                zca_whitening=False ,
                rotation_range=20 ,
                width_shift_range=0.2 ,
                height_shift_range=0.2 ,
                horizontal_flip=True ,
                vertical_flip=False
            )
            datagen.fit(dataset.train_images)
            self.model.fit_generator(datagen.flow(dataset.train_images , dataset.train_labels , batch_size=batch_size) ,
                                     nb_epoch=nb_epoch ,
                                     validation_data=(dataset.valid_images , dataset.valid_labels))
        self.save_model()
        self.Evaluate(dataset)

    def save_model(self , filepath='/home/zhangwei/face/myface_01.model.h5'):
        self.model.save(filepath=filepath)

    def load_mdoel(self , filepath='/home/zhangwei/face/myface.model.h5'):
        self.model = load_model(filepath=filepath)

    def Evaluate(self , dataset):
        score = self.model.evaluate(dataset.valid_images , dataset.valid_labels , verbose=1)
        print("%s:%.2f%%" % (self.model.metrics_names[1] , score[1] * 100))

if __name__ == '__main__':
    pathname = '/home/zhangwei/data/ScanKnife/'
    dataset = Dataset(pathname)
    dataset.load()
    # print(dataset.test_images.shape)
    # print(dataset.valid_labels)
    model = ModelFace()
    # model.build_model()
    model.train(dataset)