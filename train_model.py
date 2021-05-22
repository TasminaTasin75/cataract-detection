# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:38:59 2020

@author: USER
"""

import glob
import os
import joblib
import numpy as np
from numpy import mean
from numpy import std
import pandas as pd
import pickle
#from keras import Sequential
#from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier  
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score , recall_score , classification_report, confusion_matrix, roc_curve, auc
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

#import tensorflow as tf
#from keras.models import Sequential
#from keras.layers import Dense, Activation

def plot_roc_curve(fprs, tprs):
    #Plot the Receiver Operating Characteristic from a list of true positive rates and false positive rates.
    
    # Initialize useful lists + the plot axes.
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(14,10))
    
    # Plot ROC for each K-Fold + compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
    # Plot the luck line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    
    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    # Plot the standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    # Fine tune and show the plot.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate',fontsize=18)
    ax.set_ylabel('True Positive Rate',fontsize=18)
    ax.set_title('Receiver operating characteristic',fontsize=18)
    ax.legend(loc="lower right",fontsize=12)
    plt.savefig("ROC1.png")
    plt.show()
    
    return (f, ax)

def compute_roc_auc(index):
    y_predict = clf.predict_proba(X[index])[:,1]
    fpr, tpr, thresholds = roc_curve(y[index], y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score






RANDOM_STATE= 31

cat_feat = "features\\cataract_features"
non_cat_feat = "features\\non_cataract_features"

samples = []
labels =[]

# Get positive samples
for feat_path in glob.glob(os.path.join(cat_feat, '*.feat')):
    x = joblib.load(feat_path)
    samples.append(x)
    labels.append(1)
    
    #samples.append(np.array(x[0:62]))
# Get negative samples
for feat_path in glob.glob(os.path.join(non_cat_feat, '*.feat')):
    x = joblib.load(feat_path)
    samples.append(x)
    labels.append(0)
    #samples.append(np.array(x[0:62]))

sample_data_df = pd.DataFrame()
sample_label_df = pd.DataFrame()
sample_data_df["Sample"]=pd.Series(samples)
sample_label_df["Label"] = pd.Series(labels)

sample_data_df.sample(frac=1)



sample_data = sample_data_df["Sample"];

sample_data_reshaped=[]
print(len(sample_data))
for i in sample_data:
    sample_data_reshaped.append(i)
        
    
sample_data_reshaped = np.array(sample_data_reshaped)


sample_data_reshaped.reshape(len(sample_data_reshaped),7, 1)
#final_sample_data = sample_data_reshaped.reshape(len(sample_data_reshaped),-1)#reshaping 3d to 2d array
#final_sample_data = sample_data_reshaped.reshape(len(sample_data_reshaped),4,1)
#sample_label = sample_df["Label"].values

sample_label = sample_label_df["Label"]

X = sample_data_reshaped;
y = sample_label;

X_train, X_test, y_train, y_test = train_test_split(sample_data_reshaped, sample_label, test_size=0.25, random_state=42)

"""
clf = Sequential()
#First Hidden Layer
clf.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=7))
#Second  Hidden Layer
#clf.add(Dense(4, activation='relu', kernel_initializer='random_normal'))

#Output Layer
clf.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))


clf.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])




clf.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=4, epochs=100,shuffle=True)
_, eval_model=clf.evaluate(X_train, y_train)

"""



#clf = KNeighborsClassifier(n_neighbors =1)
#clf.fit(X_train, y_train)
#model_name='KNN.model'
 #clf = LinearSVC(random_state=RANDOM_STATE)
#clf.fit(X_train,y_train)
clf = RandomForestClassifier(n_estimators = 30)
#cv = KFold(n_splits=10, random_state=1, shuffle=True)
clf.fit(X_train,y_train)
model_name = 'random_forest.model'
#clf.save(model_name)
with open(model_name, 'wb') as file:
    pickle.dump(clf, file)
eval_model=clf.score(X_train, y_train)
predictions = clf.predict(X_test)
#predictions = clf.predict_classes(X_test)
print(classification_report(y_test,predictions))

print("Model Accuracy ",eval_model*100)
acc=accuracy_score(y_test, predictions)
print("Accuracy ",acc*100)

print("Test Set accuracy: ",clf.score(X_test, y_test)*100)
print("Train Set accuracy: ",clf.score(X_train, y_train)*100)

#scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
#print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


cv = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
results = pd.DataFrame(columns=['training_score', 'test_score'])
fprs, tprs, scores = [], [], []
    
for (train, test), i in zip(cv.split(X, y), range(5)):
    clf.fit(X[train], y[train])
    _, _, auc_score_train = compute_roc_auc(train)
    fpr, tpr, auc_score = compute_roc_auc(test)
    scores.append((auc_score_train, auc_score))
    fprs.append(fpr)
    tprs.append(tpr)

plot_roc_curve(fprs, tprs);
pd.DataFrame(scores, columns=['AUC Train', 'AUC Test'])

#scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
#print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))



cf_matrix=confusion_matrix(y_test,predictions)
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
print(tn, fp, fn, tp)

group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
#labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_counts,group_percentages)]
#labels = np.asarray(labels).reshape(2,2)
#sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristics')
plt.legend(loc="lower left",fontsize="x-large")
plt.show()
    