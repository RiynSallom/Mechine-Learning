# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:03:42 2022

@author: Rayan
"""
import joblib
import json
import pandas as pd
import numpy as np
import cv2
import matplotlib 
from matplotlib import pyplot as plt
import seaborn as sn
import os
import shutil
import pywt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
face_cascade=cv2.CascadeClassifier(".\\opencv\haar-cascade-files-master\\haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier(".\\opencv\haar-cascade-files-master\\haarcascade_eye.xml")
# img=cv2.imread(".\\test_image\\d.jpg")
# print(img.shape)
# plt.imshow(img)
# plt.show()
# # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # print(gray.shape)
# # print(gray)
# # plt.imshow(gray)
# # plt.show()
# faces=face_cascade.detectMultiScale(img,1.3,5)
# print(faces)
# (x,y,w,h)=faces[0]
# print(x,y,w,h)
# face_img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# plt.imshow(face_img)
# cv2.destroyAllWindows()
# for (x,y,w,h) in faces:
#     face_img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_color=face_img[y:y+h,x:x+w]
#     eyes=eye_cascade.detectMultiScale(roi_color)
#     print(len(eyes))
#     for (ex,ey,ew,eh) in eyes:
#         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
# plt.figure()
# plt.imshow(face_img, cmap="gray")
# plt.show()

#-----------------------DATA Cleaning-----------------------------------#
def get_cropped_if_2_eyes(image_path):
    img=cv2.imread(image_path)
    #gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(img,1.3,5)
    for (x,y,w,h) in faces:
        face_img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray=gray[y:y+h,x:x+w]
        roi_color=face_img[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_color)
        if len(eyes)>=2 and len(eyes)%2==0:
            return roi_color
path_to_data=".\\dataset\\"
path_to_cr_data=".\\dataset\\cropped\\"
img_dirs=[]
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)
print(img_dirs)
if os.path.exists(path_to_cr_data):
    shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)
cropped_image_dires=[]
celabrity_file_names_dict={}
for img_dir in img_dirs:
        count=1
        celabrity_name=img_dir.split('\\')[-1]
        if celabrity_name!='cropped':
            celabrity_file_names_dict[celabrity_name]=[]
            for entry in os.scandir(img_dir):
                roi_color=get_cropped_if_2_eyes(entry.path)
                if roi_color is not None :
                    cropped_folder=path_to_cr_data+celabrity_name
                    if not os.path.exists(cropped_folder):
                        os.makedirs(cropped_folder)
                        cropped_image_dires.append(cropped_folder)
                        print("generated cropped image in folder:  ",cropped_folder)
                    cropped_file_name=celabrity_name + str(count)+".png"
                    count+=1
                    cropped_file_path=cropped_folder+"\\"+cropped_file_name
                    cv2.imwrite(cropped_file_path,roi_color)
                    celabrity_file_names_dict[celabrity_name].append(cropped_file_path)


for img_dir in cropped_image_dires:
         celabrity_name=img_dir.split('\\')[-1]
         file_list=[]
         for entry in os.scandir(img_dir):
             print(entry.path)
             file_list.append(entry.path)
         celabrity_file_names_dict[celabrity_name]=file_list
print(celabrity_file_names_dict)
#--------------------------END DATA CLEANING-------------------------------#
    
#--------------------------Feature Engineerirng----------------------------#
print(celabrity_file_names_dict)
def w2d(img,mode='haar',level=1):
    imArray=img
    imArray=cv2.cvtColor(imArray,cv2.COLOR_RGB2GRAY)
    imArray=np.float32(imArray)
    imArray=imArray/255
    coeffs=pywt.wavedec2(imArray, mode,level=level)
    coeffs_H=list(coeffs)
    coeffs_H[0]*=0
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H*=25
    imArray_H=np.uint8(imArray_H)
    return imArray_H
count=0
class_dict={}
for key in celabrity_file_names_dict.keys():
    
        class_dict[key]=count
        count+=1
print(class_dict)
print(celabrity_file_names_dict.keys())
print(celabrity_file_names_dict)
x=[]
y=[]
for name_plyer,trining_file in celabrity_file_names_dict.items():
    for trining_image in trining_file:
        img=cv2.imread(trining_image)
        if img is None:
            continue
        scalled_row_img=cv2.resize(img,(32,32))
        img_har=w2d(img,'db1',5)
        scalled_row_img_har=cv2.resize(img_har,(32,32))
        comblaned=np.vstack((scalled_row_img.reshape(32*32*3,1),scalled_row_img_har.reshape(32*32,1)))
        x.append(comblaned)
        y.append(class_dict[name_plyer])
print(len(x[0]))
x=np.array(x).reshape(len(x),4096).astype(float)
print(x.shape)
print(celabrity_file_names_dict)
#--------------------------END Feature Engineerirng----------------------------#

#--------------------------biuld model and training------------------------------#

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
pipe=Pipeline([('scaler',StandardScaler()),('svc',SVC(kernel='rbf', C=10))])
pipe.fit(X_train,y_train)
print(pipe.score(X_test, y_test))
print(len(X_test))
print(classification_report(y_test,pipe.predict(X_test)))

model_params={
    'svm':{
        
        'model':svm.SVC(gamma='auto',probability=True),
        'params':{
            'svc__C':[1,10,100,1000],
            'svc__kernel':['rbf','linear']
            }
        },
    'random_forest':{
        
        'model':RandomForestClassifier(),
        'params':{
            'randomforestclassifier__n_estimators':[1,5,10],
            
            }
        },
    'logistic_regression':{
        
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params':{
            'logisticregression__C':[1,5,10],
            
            }
        },
    
    
    }
print(class_dict)
scores=[]
best_estimators={}
for algo,mp in model_params.items():
    pipe=make_pipeline(StandardScaler(),mp['model'])
    clf=GridSearchCV(pipe, mp['params'],cv=5,return_train_score=False)
    clf.fit(X_train,y_train)
    scores.append({
        'model':algo,
        'best_scores':clf.best_score_,
        'best_params':clf.best_params_
        
        })
    best_estimators[algo]=clf.best_estimator_
df=pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(df)
print(best_estimators['logistic_regression'].score(X_test,y_test))
best_clf=best_estimators['logistic_regression']

cm=confusion_matrix(y_test, best_clf.predict(X_test))
print(cm)
print(class_dict)

plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
#save model at pkl format
joblib.dump(best_clf,'saved_model.pkl')
with open('class_dictionary.json','w') as f:
    f.write(json.dumps(class_dict))