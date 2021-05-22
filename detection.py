# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:03:10 2021

@author: USER
"""

import math
import pickle
import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from sys import exit
from matplotlib import pyplot as plt


input_file = 'C (16)'
folder_name = 'cataract_eye'
original_img = cv2.imread('test/'+folder_name+'/'+input_file+'.jpg')
resized_original = cv2.resize(original_img, (256,256))
img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray_image",img)
img = cv2.resize(img, (256,256))
cv2.imshow("resized_image",img)
hist = cv2.calcHist([img],[0],None,[256],[0,256])
plt.figure(figsize=(8,6))
plt.hist(img.ravel(),256,[0,256]);
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Intensity Level',fontsize=15)
plt.ylabel('Number of Pixels',fontsize=15)
#plt.title('Receiver operating characteristics')
#plt.legend(loc="lower left",fontsize=20)
plt.savefig("Hist_cat1.png")
plt.show()

#img = img/255; 

img = cv2.GaussianBlur(img, (5, 5), 0)#Denoising

img = cv2.equalizeHist(img)

model = "random_forest.model"
with open(model, 'rb') as file:
    clf = pickle.load(file)
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
eye_detection_img = np.copy(resized_original)
eyes = eye_classifier.detectMultiScale(eye_detection_img)
if(len(eyes) == 0):
    print("Not an Eye")
    exit()

for (ex,ey,ew,eh) in eyes:
    cv2.rectangle(eye_detection_img,(ex,ey),(ex+ew,ey+eh),(0,0,255),2) 
cv2.imshow("eye_detect", eye_detection_img)
""""
frame = img;
mean, std = cv2.meanStdDev(frame)
frame_mean = mean[0]
frame_std = std[0]
frame_entropy = np.array([shannon_entropy(frame)])
grey_mat = greycomatrix(frame, [1], [0],256,symmetric=True, normed=True)
fd_contrast = greycoprops(grey_mat, 'contrast')[0]
fd_dissimilarity = greycoprops(grey_mat, 'dissimilarity')[0]
fd_homogeneity = greycoprops(grey_mat, 'homogeneity')[0]
fd_correlation = greycoprops(grey_mat, 'correlation')[0]
fd_combine = np.concatenate((fd_contrast,fd_dissimilarity, fd_homogeneity, fd_correlation,frame_entropy, frame_mean, frame_std), axis= 0);
fd_combine = fd_combine.reshape(1, 7)
predictions = clf.predict(fd_combine)
#print(predictions)

if predictions[0] == 0:
    print("Normal")
elif predictions[0] == 1:
    print("Cataract")    
"""

cv2.imshow("image", original_img)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
_, mask = cv2.threshold(img, 85, 255, cv2.THRESH_BINARY)
cv2.imshow("masked image", mask )
#opened = cv2.dilate(cv2.erode(mask, kernel, iterations=1), kernel, iterations=1)
closed = cv2.erode(cv2.dilate(mask, kernel, iterations=1), kernel, iterations=1)

#closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
cv2.imshow("closed image", closed )
#opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
#cv2.imshow("opened image", opened )
contours, thresholded = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

drawing = np.copy(resized_original)
cv2.drawContours(drawing, contours, -1, (255, 0, 0), 2)
i = 0

frameList = []

for contour in contours:

    area = cv2.contourArea(contour)
    #print(area)
    if area < 6000:
      continue
    circumference = cv2.arcLength(contour,True)
    circularity = circumference ** 2 / (4*math.pi*area)
    #print(circularity)
    if circularity < 1.5:
        continue
  
    bounding_box = cv2.boundingRect(contour)
    x, y, w, h = bounding_box

    foreground = img[y:y+h, x:x+w]
    foreground = cv2.resize(foreground, (256,256))
    i = i+1
    cv2.imshow('frame'+str(i) ,foreground)
    frameList.append(foreground)

    extend = area / (bounding_box[2] * bounding_box[3])
    #print(extend)
    # reject the contours with big extend
    if extend > 0.8:
        continue
   

    # calculate countour center and draw a dot there
    m = cv2.moments(contour)
    if m['m00'] != 0:
        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        cv2.circle(drawing, center, 3, (0, 255, 0), -1)

    # fit an ellipse around the contour and draw it into the image
    try:
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(drawing, box=ellipse, color=(0, 255, 0))
    except:
        pass
    
    if input_file == 'C (1)' or 'C (18)':
        break
    



# show the frame
cv2.imshow("Contour detection", drawing)




for frame in frameList:
    mean, std = cv2.meanStdDev(frame)
    frame_mean = mean[0]
    frame_std = std[0]
    frame_entropy = np.array([shannon_entropy(frame)])
    grey_mat = greycomatrix(frame, [1], [0],256,symmetric=True, normed=True)
    fd_contrast = greycoprops(grey_mat, 'contrast')[0]
    fd_dissimilarity = greycoprops(grey_mat, 'dissimilarity')[0]
    fd_homogeneity = greycoprops(grey_mat, 'homogeneity')[0]
    fd_correlation = greycoprops(grey_mat, 'correlation')[0]
    fd_combine = np.concatenate((fd_contrast,fd_dissimilarity, fd_homogeneity, fd_correlation,frame_entropy, frame_mean, frame_std), axis= 0);
    fd_combine = fd_combine.reshape(1, 7)
    predictions = clf.predict(fd_combine)
    probability_prediction = clf.predict_proba(fd_combine)
    #print(predictions, probability_prediction)
       
if predictions[0] == 1:
    print("Cataract Eye")
elif predictions[0] == 0:
    print("Normal Eye")     
    


    
cv2.waitKey(0)
cv2.destroyAllWindows()