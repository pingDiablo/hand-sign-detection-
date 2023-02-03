import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time


cap = cv2.VideoCapture('http://192.168.137.107:8080/video')
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 25
imgSize = 700
counter = 0
labels = ["Hello","Good"]

folder = 'data/'



while True:
    success, img = cap.read()
    hands = detector.findHands(img, draw = False)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        imgCropShape = imgCrop.shape


        aspectRatio = h/w
        if aspectRatio>1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgRezise = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgRezise.shape
            wGap = math.ceil((imgSize -wCal)/2)

            imgWhite[:, wGap:wCal + wGap] = imgRezise
            prediction, index = classifier.getPrediction(img)
            print(prediction, index)


        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgRezise = cv2.resize(imgCrop, (imgSize,hCal))
            imgResizeShape = imgRezise.shape
            hGap = math.ceil((imgSize - hCal) / 2)

            imgWhite[ hGap:hCal + hGap, :] = imgRezise







        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)





    cv2.imshow("Images", img)
    key = cv2.waitKey(1)









