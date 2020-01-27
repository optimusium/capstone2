# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 19:48:49 2019

@author: boonping
"""

import cv2
import os,sys
import logging as log
import datetime as dt
from time import sleep
import numpy as np
from matplotlib import pyplot as plt

'''
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
'''

from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten,Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D,MaxPooling2D,UpSampling2D
from tensorflow.keras.layers import add,Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical,plot_model
#from tensorflow.keras.datasets import cifar10
from tensorflow.keras import optimizers
from tensorflow.keras import backend
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import IPython
from scipy import ndimage
from scipy.ndimage.interpolation import shift
from numpy import savetxt,loadtxt
#savetxt('data.csv', data, delimiter=',')
#data = loadtxt('data.csv', delimiter=',')
import gc
from skimage.transform import resize


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def grayplt(img,title=''):
    '''
    plt.axis('off')
    if np.size(img.shape) == 3:
        plt.imshow(img[:,:,0],cmap='gray',vmin=0,vmax=1)
    else:
        plt.imshow(img,cmap='gray',vmin=0,vmax=1)
    plt.title(title, fontproperties=prop)
    '''
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    


    # Show the image
    if np.size(img.shape) == 3:
        ax.imshow(img[:,:,0],cmap='hot',vmin=0,vmax=1)
    else:
        ax.imshow(img,cmap='hot',vmin=0,vmax=1)
   
    plt.show()

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)    


model = load_model('facenet/facenet_keras.h5')
model.summary()
print(model.inputs)
print(model.outputs)

model.load_weights("facenet/facenet_keras_weights.h5")

images = ['frame1.jpg','frame5.jpg','frame3.jpg','frame4.jpg','frame2.jpg']
#images = ['frame2.jpg']
#p2 = 'image2/frame3.jpg'
#a=np.array([23,12,15])
#print( a[a<16].size )
#raise

os.popen("del *merged_representation*")
jjj=-1
for img in images: #def preprocess_image(img):
    jjj+=1
    imgs =np.array([])

    imag=cv2.imread(img)
    training=np.array([])
    #training.append(img.tolist())

    res = cv2.resize(imag,(160, 160), interpolation = cv2.INTER_CUBIC)
    grayplt(res/255)
    imgs=res #np.expand_dims(res,axis=0)
    print(imgs.shape)
    '''
    res=adjust_gamma(res, gamma=1.2)
    imgs=np.append(imgs,res,axis=0) 
    print(imgs.shape)
    '''
    
    '''
    res = cv2.resize(imag,(160, 160), interpolation = cv2.INTER_CUBIC)
    res = cv2.rectangle(res, (0, 110), (160, 160), (0, 0, 0), -1)
    grayplt(res/255)
    imgs=np.append(imgs,res,axis=0) 
    print(imgs.shape)
    res=adjust_gamma(res, gamma=1.2)
    imgs=np.append(imgs,res,axis=0) 
    print(imgs.shape)
    
    
    res = cv2.resize(imag,(160, 160), interpolation = cv2.INTER_CUBIC)
    res = cv2.rectangle(res, (0, 0), (160, 110), (0, 0, 0), -1)
    grayplt(res/255)
    imgs=np.append(imgs,res,axis=0) 
    print(imgs.shape)
    res=adjust_gamma(res, gamma=1.2)
    imgs=np.append(imgs,res,axis=0) 
    print(imgs.shape)
    
    
    res = cv2.resize(imag,(160, 160), interpolation = cv2.INTER_CUBIC)
    res = cv2.rectangle(res, (0, 0), (75, 110), (0, 0, 0), -1)
    grayplt(res/255)
    imgs=np.append(imgs,res,axis=0) 
    print(imgs.shape)
    res=adjust_gamma(res, gamma=1.2)
    imgs=np.append(imgs,res,axis=0) 
    print(imgs.shape)
    

    res = cv2.resize(imag,(160, 160), interpolation = cv2.INTER_CUBIC)
    res = cv2.rectangle(res, (80, 0), (160, 110), (0, 0, 0), -1)
    print(res.shape)
    grayplt(res/255)
    imgs=np.append(imgs,res,axis=0) 
    print(imgs.shape)
    res=adjust_gamma(res, gamma=1.2)
    imgs=np.append(imgs,res,axis=0) 
    print(imgs.shape)
    '''
    imgs=imgs.reshape(int(imgs.shape[0]/160),160,160,3)
    print(imgs.shape)
    
    
    #raise
    
    #res=resize(imag,(160,160))
    ##print(res.shape)
    #raise
    #im2=res
    ##############################
    
    
       
    
    '''
    grayplt(res/255)
    cascPath = "haarcascade_eye_tree_eyeglasses.xml"
    #cascPath = "haarcascade_mcs_mouth.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    log.basicConfig(filename='webcam.log',level=log.INFO)
    gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        resized = cv2.resize(res[y:y+h,x:x+w], (160,160), interpolation = cv2.INTER_AREA)
    grayplt(resized/255)
    raise
    '''
    ##############################   
        
    for iii in range(imgs.shape[0]):
        #print(img1)
        img_temp=imgs[iii]/255
        
        #cv2.imshow('frame', adjusted)
        img=imgs[iii]
        print("9981",imgs[iii].shape)
        grayplt(img/255)
        #print(img[140][25])
        #print(img[25][140])


        #continue
        ###############
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #print(hsv[140][25])
        #print(hsv[25][140])
         
        # define range of blue color in HSV
        lower_blue= np.array([0,10,45])
        lower_blue= np.array([0,10,45])
        upper_blue = np.array([60,180,255])
        upper_blue = np.array([180,180,254])
        
            
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img,img, mask= mask)
        
        imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(255-imgray, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[4]
        cv2.drawContours(im2, [cnt], 0, (255,255,255), 3)
        im2=255-im2    
        img_temp2=np.expand_dims(img_temp,axis=0)
        #grayplt(img_temp2[0])
        
        img=np.expand_dims(img,axis=0)/255
        res=np.expand_dims(res,axis=0)/255
        grayplt(im2)
        im2=im2/255
        im3=np.fliplr(im2)
        im4=np.flipud(im2)
        im5=np.fliplr(im4)
        #print(im2[im2<0.1].size,160*160*3*0.95)
        #print(im2[im2>0.1].size,160*160*3*0.95)
        
        for i in range(15):
            im2=ndimage.maximum_filter(im2, size=2)
            im3=ndimage.maximum_filter(im3, size=2)
            im4=ndimage.maximum_filter(im4, size=2)
            im5=ndimage.maximum_filter(im5, size=2)
            
            im2=ndimage.maximum_filter(im2, size=2)
            im3=ndimage.maximum_filter(im3, size=2)
            im4=ndimage.maximum_filter(im4, size=2)
            im5=ndimage.maximum_filter(im5, size=2)
            
            #im2=scipy.ndimage.gaussian_filter(im2, sigma=1.1)
            #im2=ndimage.minimum_filter(im2, size=2)
            #im3=ndimage.minimum_filter(im3, size=2)
            #im4=ndimage.minimum_filter(im4, size=2)
            #im5=ndimage.minimum_filter(im5, size=2)
            #print(im2[im2<0.1].size,160*160*3*0.95)
            #print(im2[im2>0.1].size,160*160*3*0.95)
        
        im3=np.fliplr(im3)
        im4=np.flipud(im4)
        im5=np.fliplr(im5)
        im5=np.flipud(im5)
        #grayplt(im2)
        #grayplt(im3)
        #grayplt(im4)
        #grayplt(im5)
        
        #raise
        im2=im2*im3*im4*im5
        print(im2[im2<0.1].size,160*160*3*0.95)
        print(im2[im2>0.1].size,160*160*3*0.95)
        
        img2 = np.zeros( ( np.array(im2).shape[0], np.array(im2).shape[1], 3 ) )
        img2[:,:,0] = im2 # same value in each channel
        img2[:,:,1] = im2
        img2[:,:,2] = im2
        
        im22=img2
        grayplt(im22)
        #raise
        print(im22[im22<0.1].size,160*160*3*0.95)
        print(im22[im22>0.1].size,160*160*3*0.95)
        #raise
        if (im22[im22<0.1].size)>160*160*3*0.95:
            print("skip")
            continue
        im22=img2*img_temp
        print("im22")
        #grayplt(im22)
        im2=im22*255
        #print(im2.shape)
        
        res5=im2
        
        #grayplt(res5/255)
        
    
        res=np.expand_dims(im2,axis=0)
        training=np.append(training,res)
        
    
    
        for sc in range(140,160,10):
            print("999:",sc)
            #res7 = cv2.resize(res5,(sc, sc), interpolation = cv2.INTER_CUBIC)
            res7=resize(res5,(sc,sc))
            sc1=160-sc
            sc1/=2
            sc1=int(sc1)
            sc2=80-sc1
            #print(sc1)
            res1=np.zeros((sc1,sc1,3))
            res2=np.zeros((160,sc1,3)) #np.concatenate((res1,res1,res1,res1))
            res3=np.zeros((sc1,sc2*2,3)) #np.concatenate((res1,res1),axis=1)
            #print(res.shape)
            #print(res2.shape)
            #print(res3.shape)  
            #print(res5.shape) 
            res4=np.concatenate((res3,res7,res3))
            res6=np.concatenate((res2,res4,res2),axis=1)
        
            #training.append(res.tolist())
            training=np.append(training,res6)
            #grayplt(res6/255)
            ##############
            
            
            
            for ang in [-45,-30,-15,0,15,30,45]:
                img = ndimage.rotate(res6, ang, mode='nearest')
                #print(img.shape)
                trim1=(img.shape[0]-160)/(2)
                trim1=int(trim1)
                trim2=(img.shape[1]-160)/(2)
                trim2=int(trim2)
                res1=img[trim1:trim1+160,trim2:trim2+160]
                training=np.append(training,res1)    
    
                shi=20 #int( 30-(sc-80)/2 )
                for sh in [-20,-10,0,10,20]: #range(-shi,shi,10):
                    for sh2 in [-20,-10,0,10,20]: #range(-shi,shi,10):
                        res9 = np.roll(res1, sh, axis=0)
                        res9 = np.roll(res9, sh2, axis=1)
                        grayplt(res9/255)
                        training=np.append(training,res9)
                        
            print("shape:",training.shape)
            training=training.reshape( int(training.shape[0]/76800),160,160,3)
            
            img1_representation = model.predict(training)
            #savetxt('img%i_representation_%s_%s.csv' % (jjj,iii,sc), img1_representation, delimiter=',')
            with open('img%i_merged_representation_%s.csv' % (jjj,iii), "ab") as f:
                savetxt(f, img1_representation, delimiter=',')
            
            training=np.array([])
            res5=im2
            res=np.expand_dims(im2,axis=0)
            training=np.append(training,res)
    
        for sc in range(180,220,20):
            
            #res1 = cv2.resize(res5,(sc, sc), interpolation = cv2.INTER_CUBIC)
            res1=resize(res5,(sc,sc))
            sc1=(sc-160)/2
            sc1=int(sc1)
            res1=res1[sc1:sc1+160,sc1:sc1+160]
            print("998",sc)
            grayplt(res1/255)
        
            #training.append(res.tolist())
            training=np.append(training,res1)
            for ang in [-45,-30,-15,0,15,30,45]:
                img = ndimage.rotate(res1, ang, mode='nearest')
                #print(img.shape)
                trim1=(img.shape[0]-160)/(2)
                trim1=int(trim1)
                trim2=(img.shape[1]-160)/(2)
                trim2=int(trim2)
                res2=img[trim1:trim1+160,trim2:trim2+160]
                training=np.append(training,res2)    
    
                shi=30 #int( 30-(sc-80)/2 )
                for sh in [-20,-10,0,10,20]: #range(-shi,shi,10):
                    for sh2 in [-20,-10,0,10,20]: #range(-shi,shi,10):
                        res9 = np.roll(res2, sh, axis=0)
                        res9 = np.roll(res9, sh2, axis=1)
                        training=np.append(training,res9)
    
            print("shape:",training.shape)
            training=training.reshape( int(training.shape[0]/76800),160,160,3)
            
            img1_representation = model.predict(training)
            #savetxt('img%i_representation_%s_%s.csv' % (jjj,iii,sc), img1_representation, delimiter=',')
            with open('img%i_merged_representation_%s.csv' % (jjj,iii), "ab") as ff:
                savetxt(ff, img1_representation, delimiter=',')
            
            training=np.array([])
            res5=im2
            res=np.expand_dims(im2,axis=0)
            training=np.append(training,res)
    

    #os.popen("copy img%i_representation_*.csv  img%i_merged_representation.csv" % (jjj,jjj) )        
    '''
    for i in range(training.shape[0]):
        grayplt(training[i]/255)
    raise
    '''
    #res=np.expand_dims(res,axis=0)
    #return training


#import tensorflow as tf

'''
with open('jsonmodel.json') as json_file:
    json_config = json_file.read()
model = model_from_json(json_config)

#Pre-trained OpenFace weights: https://bit.ly/2Y34cB8
model.load_weights("openface_weights.h5")
'''
#p2 = 'image2/frame2.jpg'
#preprocess_image(p1)
#raise 
#img1_representation = model.predict(preprocess_image(p1))[0,:]
#img2_representation = model.predict(preprocess_image(p2))[0,:]
#training1=preprocess_image(p1)
#savetxt('training1.csv', training1, delimiter=',')

#img2_representation = model.predict(preprocess_image(p2))



def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
 
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output
 
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    #euclidean_distance = l2_normalize(euclidean_distance )
    return euclidean_distance
 
'''
cosine = findCosineDistance(img1_representation, img2_representation)
euclidean = findEuclideanDistance(img1_representation, img2_representation)

if cosine <= 0.02:
   print("these are same")
else:
   print("these are different")
'''
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=None, max_features='auto', max_leaf_nodes=None,min_impurity_split=1e-07, min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0,n_estimators=10, n_jobs=2, oob_score=False, random_state=0,verbose=0, warm_start=False)