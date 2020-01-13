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
from tensorflow.keras import optimizers
from tensorflow.keras import backend
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import IPython
from scipy import ndimage
from scipy.ndimage.interpolation import shift
from numpy import savetxt,loadtxt

import gc
from skimage.transform import resize

from mtcnn import MTCNN

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def grayplt(img,title=''):
  
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

#Establish MTCNN detector
detector = MTCNN()
#Image List
images = ['frame1.jpg','frame5.jpg','frame3.jpg','frame4.jpg','frame2.jpg','frame6.jpg','frame7.jpg','frame8.jpg','frame9.jpg','frame10.jpg','frame11.jpg','frame12.jpg','frame13.jpg','frame14.jpg','frame15.jpg']

#Load Facenet Model
model = load_model('facenet/facenet_keras.h5')
model.summary()
print(model.inputs)
print(model.outputs)

#Load Facenet Weight
model.load_weights("facenet/facenet_keras_weights.h5")

#For pre-processed images later
imags=[]

#Counters
jjjj=-1
jjj=-1
#For each image
for img in images: 
    #Increment of counter
    jjjj+=1
    #Will be performing 6 pre-processing, hence another counter increment after 6 times.
    if jjjj%6==0:
        jjj+=1
    #Raw Image
    image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    
    #MTCNN face detection
    result = detector.detect_faces(image)
    print(result)
    if result==[]: continue
    
    # Bounding Box defined by MTCNN
    bounding_box = result[0]['box']
    
    grayplt(image/255)
    #Image is reonstructed based on MTCNN bounding box
    image=image[ bounding_box[1]:bounding_box[1]+bounding_box[3] , bounding_box[0]:bounding_box[0]+bounding_box[2] ]
    #grayplt(image/255)

    #Resized image into 160,160 for facenet input
    image = cv2.resize(image,(160, 160), interpolation = cv2.INTER_CUBIC)
    result = detector.detect_faces(image)
    print(result)    
    
    keypoints = result[0]['keypoints']
    left_eye=image[keypoints['left_eye'][1]-20:keypoints['left_eye'][1]+20, keypoints['left_eye'][0]-20:keypoints['left_eye'][0]+20]
    grayplt(left_eye/255)
       
    cv2.imwrite("ivan_drawn.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    grayplt(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)/255)
    print(keypoints['left_eye'][0])
    
    #Final Image after cropping using MTCNN
    imag = cv2.resize( cv2.cvtColor(image, cv2.COLOR_RGB2BGR),(160, 160), interpolation = cv2.INTER_CUBIC)
    grayplt(imag/255)

    #Preprocessing 0: Rebounded MTCNN image. Append image into preprocessed image list
    if imags==[] : #imags.shape[0]==0:
        #print("999")
        imags=[imag]
    else:        
        imags.append(imag)

    #Preprocessing 1: Cropped image is transformed into HSV (Darken brightest, darkest V but strengthening weak V pixel)
    hsv = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)    
    h, s, v = cv2.split(hsv)
    #Preprocessing using V component. Darken brightest pixel.
    #Strengthen pixel with V in range [20,120]
    v[v<20]=0
    v[(v>20)&(v<120)]=v[(v>20)&(v<120)]*1.08
    v[(v>180)&(v<250)]=v[(v>180)&(v<250)]*0.92
    v[v>250]=245
    hsv = cv2.merge((h, s, v))
    imag2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #grayplt(imag2/255)
    #print(imags.shape)
    #imags=np.append(imags,imag2,axis=0)
    imags.append(imag2)

    #Preprocessing 2: Cropped image is transformed into HSV (Darkening using V)
    hsv = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)    
    #Preprocessing using V component. Darken brightest pixel.
    #Darken pixel with V in range [20,120]
    h, s, v = cv2.split(hsv)
    v[v<20]=0
    v[(v>20)&(v<120)]=v[(v>20)&(v<120)]*0.92
    v[(v>180)&(v<250)]=v[(v>180)&(v<250)]*0.98
    v[v>250]=250
    hsv = cv2.merge((h, s, v))
    imag2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #grayplt(imag2/255)
    #print(imags.shape)
    #imags=np.append(imags,imag2,axis=0)
    imags.append(imag2)

    #Preprocessing 3: Cropped image is transformed into BGR (Re-processing R and G where these are main human skin color components)
    b, g, r = cv2.split(imag)
    r[r<15]=0
    r[(r>15)&(r<100)]=r[(r>15)&(r<100)]*1.1
    r[(r>180)&(r<250)]=r[(r>180)&(r<250)]*0.9
    r[r>250]=250

    g[g<15]=0
    g[(g>15)&(g<100)]=g[(g>15)&(g<100)]*1.1
    g[(g>180)&(g<250)]=g[(g>180)&(g<250)]*0.9
    g[g>250]=250

    imag2 = cv2.merge((b, g, r))
    #grayplt(imag2/255)
    #print(imags.shape)
    #imags=np.append(imags,imag2,axis=0)
    imags.append(imag2)
    
    #Preprocessing 4: Cropped image is transformed using gamma 1.2
    imag2=adjust_gamma(imag, gamma=1.2)
    imags.append(imag2)
    #Preprocessing 5: Cropped image is transformed using gamma 0.8
    imag2=adjust_gamma(imag, gamma=0.8)
    imags.append(imag2)
    
#delete existing file consits of facenet prediction    
os.popen("del *merged_representation*")    
    
#counters    
jjj=-1
jjjj=-1
#for each pre-processed images
for imag in imags: 
    #increment of counter
    jjjj+=1
    #Will be performing 6 pre-processing, hence another counter increment after 6 times.
    if jjjj%6==0:
        jjj+=1

    #resize images
    imag = cv2.resize(imag,(160, 160), interpolation = cv2.INTER_CUBIC)
    
    #Analyzing face color, to get exact pixels with face color. 
    #After getting these pixels, change it to white. Non face color in black.
    #Then using flooding method, fill the black pixels within white contour.
    #Then bitwise AND with MTCNN pre-processed images.
    #This will remove obvious nackgrounds.

    #HSV preprocessing. Darken bright pixels to prevent over-exposure
    hsv = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)    
    h, s, v = cv2.split(hsv)
    
    lim = 250
    v[v > lim] = v[v > lim]*0.95
    lim = 220
    v[v > lim] = v[v > lim]*0.95
    lim = 200
    v[v > lim] = v[v > lim]*0.95
    lim = 150
    v[v > lim] = v[v > lim]*0.95
    lim=20
    v[v < lim] = 0

    hsv = cv2.merge((h, s, v))        
    #print (hsv[30][80])
    #print (hsv[80][30])
    
    res=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    grayplt(res/255)
    
    #Getting pixels with face color. White is pixel with face color features while zero is non-face pixels.
    #Face, should have a significant contrast in H 
    #Face, should have low S
    #Face may be over-exposured, so V had been pre-processed.
    #Face color is having significant G, R but less B.
    h, s, v = cv2.split(hsv)
    print("h")
    l=cv2.resize(h,(160, 160), interpolation = cv2.INTER_CUBIC)
    l=l[0:160,0:120]
    l[l>75]=179
    l[l<=75]=0
    if l[l==0].shape[0] > l[l==179].shape[0]:
        l[l==0]=255
        l[l==179]=0
    else:
        l[l==179]=255
    l[l==0]=100
        
    
    #grayplt(l/180)
    
    ri=cv2.resize(h,(160, 160), interpolation = cv2.INTER_CUBIC)
    ri=ri[0:160,40:160]
    ri[ri>75]=179
    ri[ri<=75]=0
    if ri[ri==0].shape[0] > ri[ri==179].shape[0]:
        ri[ri==0]=255
        ri[ri==179]=0
    else:
        ri[ri==179]=255
    ri[ri==0]=100
    
    com=np.append(l[0:160,0:120],ri[0:160,80:120],axis=1)
    grayplt(com/255)
    
    
    #grayplt(h/180)
    print("s")
    l=cv2.resize(s,(160, 160), interpolation = cv2.INTER_CUBIC)
    l[l>185]=255
    l[l<=185]=0      
    com2=cv2.resize(l,(160, 160), interpolation = cv2.INTER_CUBIC)
    grayplt(com2/255)
    #grayplt(s/255)
    print("v")
    v[v<100]=0
    v[(v<120)&(v!=0)]-=30
    v[(v<210)&(v!=0)]-=20
    v[v>235]=0
    grayplt(v/255)

    b, g, r = cv2.split(res)
    print("b")
    #l=cv2.resize(b,(160, 160), interpolation = cv2.INTER_CUBIC)
    b[b>230]=255
    b[b<=230]=0      
    grayplt(b/255)
    
    print("g")
    g[g<60]=1
    g[g>=60]=254
    g[g==1]=255
    g[g==254]=0
    grayplt(g/255)
    
    print("r")
    r[r<40]=1
    r[r>=40]=254
    r[r==1]=255
    r[r==254]=0
    grayplt(r/255)
    
    print("combined")
    fin=(com/255)*(v/255)-(b/255)-(com2/255)-(r/255)-(g/255)
    fin[fin<0.02]=0
    l=fin[0:160,0:10]
    l[l<0.33]=0

    fin[(fin>0.2)&(fin<0.3)]+=0.3
    fin[fin>0.3]+=0.3
    fin[fin>0.9]=1
    
    fin[fin>0.12]=1
    fin[fin<=0.12]=0
    
    fin=fin*255
    grayplt( fin/255 )
    
    #Use a series of maxand min filter to perform flooding on black pixels surrounded by white pixels
    if 1:
        
        im2=fin/255
        im3=np.fliplr(im2)
        im4=np.flipud(im2)
        im5=np.fliplr(im4)
        
        for i in range(15):
            im2=ndimage.maximum_filter(im2, size=2)
            im3=ndimage.maximum_filter(im3, size=2)
            im4=ndimage.maximum_filter(im4, size=2)
            im5=ndimage.maximum_filter(im5, size=2)
            
            im2=ndimage.maximum_filter(im2, size=2)
            im3=ndimage.maximum_filter(im3, size=2)
            im4=ndimage.maximum_filter(im4, size=2)
            im5=ndimage.maximum_filter(im5, size=2)

        for i in range(3):
            im2=ndimage.minimum_filter(im2, size=2)
            im3=ndimage.minimum_filter(im3, size=2)
            im4=ndimage.minimum_filter(im4, size=2)
            im5=ndimage.minimum_filter(im5, size=2)

        
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
        #At this point im22 represnts face pixel. White should be face pixels/
        
        img2 = np.zeros( ( np.array(im2).shape[0], np.array(im2).shape[1], 3 ) )
        img2[:,:,0] = im2 # same value in each channel
        img2[:,:,1] = im2
        img2[:,:,2] = im2
        #img2 is 3-dimensional im22
        
        im22=img2
        grayplt(im22*imag/255)
        res5=im22*imag
        im22=im22*imag
        #Both res5 and im22 are BITWISE AND result of img2 and MTCNN pre-processed images. (imag is MTCNN pre-processed images)
        #Both res5 and im22 are MTCNN images without bvious background.

        training=np.array([])
        res=np.expand_dims(im22,axis=0)
        #Append the final MTCNN pre-processed image without obvious ackground into facenet training list
        training=np.append(training,res)
        #print(res5.shape)
        #raise
    
        iii=0
        #Re-scaling image (zoom-out)
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
        
            #Adding scaled image into facenet training set
            training=np.append(training,res6)
            #grayplt(res6/255)
            ##############
            
            
            #For each scaled image, rotate using different angles.
            for ang in [-45,-30,-15,0,15,30,45]:
                img = ndimage.rotate(res6, ang, mode='nearest')
                #print(img.shape)
                trim1=(img.shape[0]-160)/(2)
                trim1=int(trim1)
                trim2=(img.shape[1]-160)/(2)
                trim2=int(trim2)
                res1=img[trim1:trim1+160,trim2:trim2+160]
                #Append each rotated+scaled image into training set
                training=np.append(training,res1)    
    
                #for each rotated+scaled image, perform translation
                shi=20 #int( 30-(sc-80)/2 )
                for sh in [-20,0,20]: #range(-shi,shi,10):
                    for sh2 in [-20,0,20]: #range(-shi,shi,10):
                        res9 = np.roll(res1, sh, axis=0)
                        res9 = np.roll(res9, sh2, axis=1)
                        grayplt(res9/255)
                        #Append each rotated+scaled+translation image into training set
                        training=np.append(training,res9)
                        
            print("shape:",training.shape)
            #Reshape the training set (since previous append was done pixel by pixel)
            training=training.reshape( int(training.shape[0]/76800),160,160,3)
            
            #facenet prediction 
            img1_representation = model.predict(training)
            #save the numpy array which consists of facenet prediction into csv file.
            with open('img%i_merged_representation.csv' % (jjj), "ab") as f:
                savetxt(f, img1_representation, delimiter=',')
            
            #Free-up training space
            training=np.array([])
            #res5=im2
            #res=np.expand_dims(im2,axis=0)
            training=np.append(training,res)
        iii=1
        #Scaling (zoom-in)
        for sc in range(180,230,10):
            
            #res1 = cv2.resize(res5,(sc, sc), interpolation = cv2.INTER_CUBIC)
            res1=resize(res5,(sc,sc))
            sc1=(sc-160)/2
            sc1=int(sc1)
            #Always scaled back to 160,160
            res1=res1[sc1:sc1+160,sc1:sc1+160]
            print("998",sc)
            grayplt(res1/255)
        
            #training.append(res.tolist())
            #Append scaled images
            training=np.append(training,res1)
            #rotation
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
                #angle
                for sh in [-20,0,20]: #range(-shi,shi,10):
                    for sh2 in [-20,0,20]: #range(-shi,shi,10):
                        res9 = np.roll(res2, sh, axis=0)
                        res9 = np.roll(res9, sh2, axis=1)
                        training=np.append(training,res9)
    
            print("shape:",training.shape)
            training=training.reshape( int(training.shape[0]/76800),160,160,3)
            
            #facenet prediction
            img1_representation = model.predict(training)
            #savetxt('img%i_representation_%s_%s.csv' % (jjj,iii,sc), img1_representation, delimiter=',')
            #save prediction by appending to file.
            with open('img%i_merged_representation.csv' % (jjj), "ab") as ff:
                savetxt(ff, img1_representation, delimiter=',')
            
            training=np.array([])
            #res5=im2
            #res=np.expand_dims(im2,axis=0)
            training=np.append(training,res)


