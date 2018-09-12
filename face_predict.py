#-*- coding:utf-8 -*-
#author:zhangwei

import cv2
from face_dl import ModelFace
from load_dataset import resize_image

MODEL_PATH = '/home/zhangwei/face/myface.model.h5'

if __name__ == '__main__':
    model = ModelFace()
    model.load_mdoel()
    imagepath = '/home/zhangwei/138439.jpg'
    image = cv2.imread(imagepath)
    image = resize_image(image)
    # cv2.imshow("image" , image)
    # cv2.imwrite("/home/zhangwei/1.jpg" , image)
    # cv2.waitKey(10)
    model = ModelFace()
    model.load_mdoel()
    model.face_predict_01(image)
