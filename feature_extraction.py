# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 13:38:21 2020

@author: USER
"""

from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
#from skimage.io import imread
import joblib
import glob
import os
import cv2
import numpy as np

cat_im_path = "augmented/cataract"
non_cat_im_path = "augmented/non_cataract"

cat_feat_path = "features\\cataract_features"
non_cat_feat_path = "features\\non_cataract_features"

model_path = "model"

if not os.path.isdir(cat_feat_path):
    os.makedirs(cat_feat_path)
    
if not os.path.isdir(non_cat_feat_path):
    os.makedirs(non_cat_feat_path)


for im_path in glob.glob(os.path.join(cat_im_path, "*")):
    #print(im_path)
    cat_image = cv2.imread(im_path, 0)
    #cat_image = cv2.resize(cat_image, (360, 480))
    cat_image = cv2.resize(cat_image, (256, 256))#Resizing
    cat_image = cv2.GaussianBlur(cat_image, (5, 5), 0)#Denoising
    cat_image =cv2.equalizeHist(cat_image)
    mean, std = cv2.meanStdDev(cat_image)
    cat_image_mean = mean[0]
    cat_image_std = std[0]
    cat_image_entropy = np.array([shannon_entropy(cat_image)])
    
    #cat_image_feats = np.array([shannon_entropy(cat_image),0, 0,0]).reshape(1,4)

    #print(cat_image_mean)
    grey_mat = greycomatrix(cat_image, [1], [0],256,symmetric=True, normed=True)
    fd_contrast = greycoprops(grey_mat, 'contrast')[0]
    fd_dissimilarity = greycoprops(grey_mat, 'dissimilarity')[0]
    fd_homogeneity = greycoprops(grey_mat, 'homogeneity')[0]
    fd_ASM = greycoprops(grey_mat, 'ASM')[0]
    fd_correlation = greycoprops(grey_mat, 'correlation')[0]
    fd_energy = greycoprops(grey_mat, 'energy')[0]

    #fd_combine = np.concatenate((fd_contrast,fd_dissimilarity, fd_homogeneity, fd_correlation, cat_image_mean, cat_image_std,  cat_image_entropy), axis= 0);
    fd_combine = np.concatenate((fd_contrast, fd_dissimilarity, fd_homogeneity, fd_correlation, cat_image_entropy,cat_image_mean,cat_image_std   ), axis= 0);
    fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
    fd_path = os.path.join(cat_feat_path, fd_name)
    
    joblib.dump(fd_combine, fd_path)
print("Cataract features saved in {}".format(cat_feat_path)) 
 
for im_path in glob.glob(os.path.join(non_cat_im_path, "*")):
    #print(im_path)
    non_cat_image = cv2.imread(im_path, 0)
    #non_cat_image = cv2.resize(non_cat_image, (360, 480))
    non_cat_image = cv2.resize(non_cat_image, (256, 256))
    non_cat_image = cv2.GaussianBlur(non_cat_image, (5, 5), 0)
    non_cat_image =cv2.equalizeHist(non_cat_image)
    mean, std = cv2.meanStdDev(non_cat_image)
    non_cat_image_mean = mean[0]
    non_cat_image_std = std[0]
    non_cat_image_entropy = np.array([shannon_entropy(non_cat_image)])
    #non_cat_image_feats = np.array([shannon_entropy(non_cat_image),0, 0,0]).reshape(1,4)
    #print(non_cat_image_mean)
    grey_mat = greycomatrix(non_cat_image, [1], [0],256,symmetric=True, normed=True)
    fd_contrast = greycoprops(grey_mat, 'contrast')[0]
    fd_dissimilarity = greycoprops(grey_mat, 'dissimilarity')[0]
    fd_homogeneity = greycoprops(grey_mat, 'homogeneity')[0]
    fd_ASM = greycoprops(grey_mat, 'ASM')[0]
    fd_correlation = greycoprops(grey_mat, 'correlation')[0]
    fd_energy = greycoprops(grey_mat, 'energy')[0]

    #fd_combine = np.concatenate((fd_contrast,fd_dissimilarity, fd_homogeneity, fd_correlation, non_cat_image_mean, non_cat_image_std, non_cat_image_entropy), axis= 0);
    fd_combine = np.concatenate((fd_contrast, fd_dissimilarity, fd_homogeneity, fd_correlation, non_cat_image_entropy,non_cat_image_mean,non_cat_image_std), axis= 0);
    fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
    fd_path = os.path.join(non_cat_feat_path, fd_name)

    joblib.dump(fd_combine, fd_path)
print("Normal features saved in {}".format(non_cat_feat_path)) 
