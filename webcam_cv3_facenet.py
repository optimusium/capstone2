import cv2
import sys
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

#import tensorflow as tf
model = load_model('facenet/facenet_keras.h5')
model.summary()
print(model.inputs)
print(model.outputs)

model.load_weights("facenet/facenet_keras_weights.h5")


def createModel():
    '''
    inputShape=(16,) #128/8
    
    inputs0      = Input(shape=inputShape)
    inputs1      = Input(shape=inputShape)
    inputs2      = Input(shape=inputShape)
    inputs3      = Input(shape=inputShape)
    inputs4      = Input(shape=inputShape)
    inputs5      = Input(shape=inputShape)
    inputs6      = Input(shape=inputShape)
    inputs7      = Input(shape=inputShape)
    
    
    x0 = Dense(8,activation="relu")(inputs0)
    x1 = Dense(8,activation="relu")(inputs1)
    x2 = Dense(8,activation="relu")(inputs2)
    x3 = Dense(8,activation="relu")(inputs3)
    x4 = Dense(8,activation="relu")(inputs4)
    x5 = Dense(8,activation="relu")(inputs5)
    x6 = Dense(8,activation="relu")(inputs6)
    x7 = Dense(8,activation="relu")(inputs7)
    
    
    x0=Reshape((16,1))(inputs0)
    x1=Reshape((16,1))(inputs1)
    x2=Reshape((16,1))(inputs2)
    x3=Reshape((16,1))(inputs3)
    x4=Reshape((16,1))(inputs4)
    x5=Reshape((16,1))(inputs5)
    x6=Reshape((16,1))(inputs6)
    x7=Reshape((16,1))(inputs7)
    
    x0=Conv1D(8,kernel_size=(8,),activation="relu")(x0)
    x1=Conv1D(8,kernel_size=(8,),activation="relu")(x1)
    x2=Conv1D(8,kernel_size=(8,),activation="relu")(x2)
    x3=Conv1D(8,kernel_size=(8,),activation="relu")(x3)
    x4=Conv1D(8,kernel_size=(8,),activation="relu")(x4)
    x5=Conv1D(8,kernel_size=(8,),activation="relu")(x5)
    x6=Conv1D(8,kernel_size=(8,),activation="relu")(x6)
    x7=Conv1D(8,kernel_size=(8,),activation="relu")(x7)
    
    
    x=concatenate([x0,x1,x2,x3,x4,x5,x6,x7])
    
    x = Dense(128,activation="relu")(x)
    x = Dense(64,activation="relu")(x)
    #x= Flatten()(x)
    x = Dense(32,activation="relu")(x)
    x = Dense(20,activation="relu")(x)
    '''
    
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

model2=createModel()
model2.summary()
modelname="facenet_network"
model2.load_weights(modelname + ".hdf5")


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

    '''
    movement=0    
    
    if prev==0:
        #prev_hist=histr
        prev_frame=frame
        prev=1
        sleep=(0.25)
        continue
    else:
        color = ('b','g','r')
        histr=[]
        for i,col in enumerate(color):
            histr = moving_average( cv2.calcHist([frame],[i],None,[256],[0,256]) )
            #print(histr)
            #histr2 = moving_average( cv2.calcHist([prev_frame],[i],None,[256],[0,256]) )
            #print(histr2)     
            #print(histr-histr2)
            #raise
            #histr3 = histr-histr2 #moving_average( cv2.calcHist([frame-prev_frame],[i],None,[256],[0,256]) )
            #histr3/=histr+1
            #print(histr3)
            #raise
            #plt.plot(histr,color = col)
            #plt.plot(histr2,color = col)
            plt.plot(histr,color = col)
            plt.xlim([0,256])
            plt.show()

        #plt.plot(histr,color = col)
        #plt.plot(histr2,color = col)
        #plt.plot(histr3,color = col)
        #plt.plot(prev_hist,color = col)
        #plt.plot(histr-prev_hist,color = col)
        #plt.xlim([0,256])
        #plt.show()
        #prev_hist=histr
        prev_frame=frame
        sleep=(1)
    '''
    '''    
    negative=frame-prev_frame
    negative=np.where(negative>240,0,negative)
    negative=np.where(negative<15,0,negative)
    '''
    
    
    #grayplt(negative)
    #grayplt(frame)
    #grayplt(prev_frame)
    prev_frame=frame
    #print( np.size(negative))
    #print( np.sum(negative>120)  )
    #sleep(1
    #print(frame)
    #print(prev_frame)
    #raise


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        resized = cv2.resize(frame[y:y+h,x:x+w], (160,160), interpolation = cv2.INTER_AREA)
        
        ######################
        img_temp=resized/255
        adjusted = adjust_gamma(resized, gamma=1.2)
        #cv2.imshow('frame', adjusted)
        img=adjusted
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
         
        # define range of blue color in HSV
        lower_blue= np.array([0,10,45])
        upper_blue = np.array([55,180,255])
        lower_blue= np.array([0,10,45])
        upper_blue = np.array([180,180,255])
        
            
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img,img, mask= mask)
        #print(res[res<0.1].shape)
        if res[res<0.1].shape[0]>0.95*(res.shape[0]*res.shape[1]*res.shape[2]):
            continue
        
        imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(255-imgray, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print("contours",contours)
        going=0
        try:
            cnt = contours[4]
        except:
            going=1
            pass
        if going==1: continue
        cv2.drawContours(im2, [cnt], 0, (255,255,255), 3)
        im2=255-im2    
        img_temp2=np.expand_dims(img_temp,axis=0)
        #grayplt(img_temp2[0])
        
        img=np.expand_dims(img,axis=0)/255
        res=np.expand_dims(res,axis=0)/255
        
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
        ######################
        
        #resized=np.expand_dims(resized,axis=0)
        #p1 = 'frame5.jpg'
        #p2 = 'image2/frame3.jpg'
        #p2 = 'image2/frame2.jpg'
         
        #img1_representation = model.predict(preprocess_image(p1))[0,:]
        img2_representation = model.predict(resized) #(preprocess_image(resized))[0,:]
        '''
        cosine = findCosineDistance(img1_representation, img2_representation)
        euclidean = findEuclideanDistance(img1_representation, img2_representation)
        
        if cosine <= 0.02:
           print("this is boonping")
        else:
           print("this is not boonping")
        '''
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
        
        if sel==0: print("not recognized")
        elif sel==1: print("JunLeng")
        elif sel==2: print("BoonPing")
        elif sel==3: print("YeongShin")
        elif sel==4: print("Francis")
                
        '''
        if np.argmax(prediction[0])==1:
           print("this is boonping")
        else:
           print("this is not boonping")
        ''' 
        

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        sleep(0.5)
        '''
        resized=np.expand_dims(resized,axis=0)/255
        print(resized.shape)
        

        
        predicts_img    = modelGo.predict(resized)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,500)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2
        grayplt(resized[0])
        print(np.argmax(predicts_img[0]))
        print(predicts_img[0])
        
        image = load_img("frame7.jpg")
        resized2=np.expand_dims(image,axis=0)/255
        predicts_img    = modelGo.predict(resized2)
        grayplt(resized2[0])
        print(np.argmax(predicts_img[0]))
        print(predicts_img[0])
        
        cv2.putText(frame,'%s' % np.argmax(predicts_img), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
        '''
        
        '''
        cv2.imwrite("frame.jpg",frame[y:y+h,x:x+w])
        #cv2.imwrite("frame.jpg",frame)
        resized = cv2.resize(frame[y:y+h,x:x+w], (200,200), interpolation = cv2.INTER_AREA)
        cv2.imwrite("frame2.jpg",resized)
        
        #raise
        '''
        

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        '''
        image = load_img("frame2.jpg")
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        aug=ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")
        print("[INFO] generating images...")
        imageGen = aug.flow(image, batch_size=1, save_to_dir=".",save_prefix="image5", save_format="jpg")
        i=0
        for image in imageGen:
            print(image)
            i+=1
            if i==100: break
        '''
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
