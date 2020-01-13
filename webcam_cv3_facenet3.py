import cv2
import sys
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
#from tensorflow.keras.datasets import cifar10
from tensorflow.keras import optimizers
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import IPython
from scipy import ndimage
from scipy.ndimage.interpolation import shift
from numpy import savetxt,loadtxt
#savetxt('data.csv', data, delimiter=',')
#data = loadtxt('data.csv', delimiter=',')
import gc
from skimage.transform import resize

import pickle
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from mtcnn import MTCNN
detector = MTCNN()

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

def preprocess_image(img):
    imag=cv2.imread(img)
    res = cv2.resize(imag,(160, 160), interpolation = cv2.INTER_CUBIC)
    res=np.expand_dims(res,axis=0)
    return res

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

#import facenet model
model = load_model('facenet/facenet_keras.h5')
model.summary()
print(model.inputs)
print(model.outputs)
#import facenet weight
model.load_weights("facenet/facenet_keras_weights.h5")

#from facenet_network.py
def createModel():
    
    inputShape=(128,)
    inputs      = Input(shape=inputShape)
    #x=Reshape((128,1))(inputs)
    x = Dense(128,activation="relu")(inputs)
    #x=Conv1D(128,kernel_size=(8,),activation="relu",padding="same")(x)
    #x=Conv1D(64,kernel_size=(4,),activation="relu",padding="same")(x)
    #x=AveragePooling1D(4)(x)
    #x = Flatten()(x)
    x = Dense(64,activation="relu")(x)
    x = Dense(32,activation="relu")(x)
    x = Dense(20,activation="relu")(x)

    outputs0 = Dense(20,activation="relu")(x)
    outputs1 = Dense(20,activation="relu")(x)
    outputs2 = Dense(20,activation="relu")(x)
    outputs3 = Dense(20,activation="relu")(x)
    
    outputs0 = Dense(2,activation="softmax")(outputs0)
    outputs1 = Dense(2,activation="softmax")(outputs1)
    outputs2 = Dense(2,activation="softmax")(outputs2)
    outputs3 = Dense(2,activation="softmax")(outputs3)
    
    model       = Model(inputs=inputs,outputs=[outputs0,outputs1,outputs2,outputs3])       
    #model       = Model(inputs=[inputs0,inputs1,inputs2,inputs3,inputs4,inputs5,inputs6,inputs7],outputs=outputs)       
    model.compile(loss='categorical_crossentropy', 
                optimizer=optimizers.Adam() ,
                metrics=['accuracy'])

    return model

#Load weight for Neural Network Classfier. From facenet_network.py
model2=createModel()
model2.summary()
modelname="facenet_network"
model2.load_weights(modelname + ".hdf5")

#Load trained SVM trained model. From svm.py
filename="svm0.sav"
model3=pickle.load(open(filename,'rb'))
filename="svm1.sav"
model4=pickle.load(open(filename,'rb'))
filename="svm2.sav"
model5=pickle.load(open(filename,'rb'))
filename="svm3.sav"
model6=pickle.load(open(filename,'rb'))

#Load HAar Wavelet file. create face cascade
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

prev=0
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #cv2.imwrite("frame.jpg",frame)
    #sleep(0.25)
    prev_frame=frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # x,y,w,h are bounds defined by Haar Wavelet
    for (x, y, w, h) in faces:
        #resized image
        resized = cv2.resize(frame[y:y+h,x:x+w], (160,160), interpolation = cv2.INTER_AREA)
        grayplt(resized/255)
        
        #Use MTCNN
        result = detector.detect_faces(resized)
        print(result)
        #If MTCNN cannot recognize the face, skip this frame.
        if result==[]: continue
        
        # Bounding box by MTCNN.
        bounding_box = result[0]['box']
        
        #REsized using MTCNN bounds.
        resized=resized[ bounding_box[1]:bounding_box[1]+bounding_box[3] , bounding_box[0]:bounding_box[0]+bounding_box[2] ]
        #resized to 160,160 again
        resized = cv2.resize(resized,(160, 160), interpolation = cv2.INTER_CUBIC)
        
        #save original image
        img_temp=resized/255

        #Getting all pixel with face color and save as another image. 
        #Remove background from image.
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)    
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

        # define range of blue color in HSV
        res=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        h, s, v = cv2.split(hsv)
        #print("h")
        #l=cv2.equalizeHist(l)
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
        
        #grayplt(ri/180)
        
        com=np.append(l[0:160,0:120],ri[0:160,80:120],axis=1)
        #grayplt(com/255)
        
        #grayplt(h/180)
        #print("s")
        l=cv2.resize(s,(160, 160), interpolation = cv2.INTER_CUBIC)
        l[l>185]=255
        l[l<=185]=0      
        com2=cv2.resize(l,(160, 160), interpolation = cv2.INTER_CUBIC)
        #grayplt(com2/255)
        #grayplt(s/255)
        #print("v")
        v[v<100]=0
        v[(v<120)&(v!=0)]-=30
        v[(v<210)&(v!=0)]-=20
        v[v>235]=0
        #grayplt(v/255)
    
        b, g, r = cv2.split(res)
        #print("b")
        #l=cv2.resize(b,(160, 160), interpolation = cv2.INTER_CUBIC)
        b[b>230]=255
        b[b<=230]=0      
        #grayplt(b/255)
        
        #grayplt(b/255)
        #g[g<50]=0
        #g[g>=50]=255
        #print("g")
        g[g<60]=1
        g[g>=60]=254
        g[g==1]=255
        g[g==254]=0
        #grayplt(g/255)
        
        #print("r")
        r[r<40]=1
        r[r>=40]=254
        r[r==1]=255
        r[r==254]=0
        #grayplt(r/255)
        
        #print("combined")
        fin=(com/255)*(v/255)-(b/255)-(com2/255)-(r/255)-(g/255)
        fin[fin<0.02]=0
        l=fin[0:160,0:10]
        l[l<0.33]=0

        #fin[fin<0.6]-=0.1
        #fin=fin*fin
        fin[(fin>0.2)&(fin<0.3)]+=0.3
        fin[fin>0.3]+=0.3
        fin[fin>0.9]=1
        
        fin[fin>0.12]=1
        fin[fin<=0.12]=0
        
        im2=fin*255
        #grayplt( fin )
        
        #9999
        im2=im2/255
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
            
            #im2=scipy.ndimage.gaussian_filter(im2, sigma=1.1)
            im2=ndimage.minimum_filter(im2, size=2)
            im2=ndimage.minimum_filter(im3, size=2)
            im2=ndimage.minimum_filter(im4, size=2)
            im2=ndimage.minimum_filter(im5, size=2)
        
        im3=np.fliplr(im3)
        im4=np.flipud(im4)
        im5=np.fliplr(im5)
        im5=np.flipud(im5)
        
        im2=im2*im3*im4*im5
        
        #print(im2[im2<0.1].shape)
        if im2[im2<0.1].shape[0]>0.95*(im2.shape[0]*im2.shape[1]):
            continue
        
        img2 = np.zeros( ( np.array(im2).shape[0], np.array(im2).shape[1], 3 ) )
        img2[:,:,0] = im2 # same value in each channel
        img2[:,:,1] = im2
        img2[:,:,2] = im2
        
        im22=img2*img_temp
        grayplt(im22)
        

        im2=im22*255
        #print(im2.shape)
        
        res5=im2

        #res=np.expand_dims(im2,axis=0)
        
        resized=np.expand_dims(im2,axis=0)
        #Removed background. Resized is the final pre-processed image
        ######################
        
        #resized=np.expand_dims(resized,axis=0)
        #p1 = 'frame5.jpg'
        #p2 = 'image2/frame3.jpg'
        #p2 = 'image2/frame2.jpg'
         
        #img1_representation = model.predict(preprocess_image(p1))[0,:]

        #Facenet prediction of image
        img2_representation = model.predict(resized) #(preprocess_image(resized))[0,:]

        #Face recognition using SVM predictions for each image.
        result2=model3.predict(img2_representation)
        result3=model4.predict(img2_representation)
        result4=model5.predict(img2_representation)
        result5=model6.predict(img2_representation)

        #print(img2_representation.shape)

        #Print SVM results.
        print("aujunleng",result2)
        print("boonping",result3)
        print("yeongshin",result4)
        print("francis",result5)
        
        #Face recognition using neural network classfier        
        prediction=model2.predict(img2_representation)
        #print(np.argmax(prediction[0]))
        print(prediction)
        
        fa=-1
        val=0
        sel=0
        for fac in range(3):
            fa+=1
            if np.argmax(prediction[fac][0])==1:
                if fa==0 and prediction[fac][0][1]>val: 
                    sel=1
                    val=prediction[fac][0][1]
                    
                if fa==1 and prediction[fac][0][1]>val: 
                    sel=2
                    val=prediction[fac][0][1]
                    
                if fa==2 and prediction[fac][0][1]>val: 
                    sel=3
                    val=prediction[fac][0][1]

                if fa==3 and prediction[fac][0][1]>val: 
                    sel=4
                    val=prediction[fac][0][1]
                    
        if val<0.55: sel=0

        #print neural network result.
        if sel==0: print("not recognized")
        elif sel==1: print("JunLeng")
        elif sel==2: print("BoonPing")
        elif sel==3: print("YeongShin")
        elif sel==4: print("Francis")

        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        sleep(0.5)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
