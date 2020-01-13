# capstone2

Use ml1p13 env
USe frameX.jpg as images.

1. webcam_cv3_capture.py 

    Using HAar wavelet transform to capture image in video. Then using MTCNN to refine image. Save image as frame.jpg
    
2. facenet_predict3.py

    This is to use images (can be from webcam_cv3_capture or any other images) to run facenet prediction.
    
    a) Use MTCNN to crop and align the face.

    b) cropped image is used for RGB/HSV analysis. The Analysis result image is used to capture all the pixels with face skin color. 
    
        The resultant mask image , with white as pixel with face color. The black holes (eyes/mouth) will be filled with cv2 flooding functions.
        
        The mask image is then bitwise AND with MTCNN processed image. This complete 1st stage of preprocessing
        
    c) Adding this image into preprocessing image list
    
    d) 1st stage pre-processed image is used for 2nd stage pre-processing.
    
       i) Darken brightenest pixel.
       
       ii) Darken all pixels
       
       iii) Scaling R and G component. (Darken cheek)
       
       iv) gamma 0.8
       
       v) gamma 1.2
       
    e) All the original images will have 6 pre-processed images after 2 pre-processing stages.
    
    f) All the images will be scaled (zoom in and out), rotated, translated to form facenet input sets.
    
    g) Use loaded model and weight to run facenet on the input set.
    
    h) save predictions of each image into 1 csv file.
    
3. svm.py

    This is to load csv file data as input, train the SVM model.
    
    4 SVM models will be created for 1st 4 images.
    
    The resultant models are saved in .sav files.
    
4. facenet_network.py

    This is to load csv file as input, train the neural network classifier.
    
    This is 1 input 4 binary output neural classifer 
    
    The resultant model are saved in hdf5 files.
    
5. webcam_cv3_facenet3.py

    This is the application file.
    
    a) Loading facenet model and weight. Loading SVM models and neural classfier model/weight in step 3.
    
    b) Webcam will capture the frame. Haar wavelet captures face image out of the frame.
    
    c) The captured frame is going thru MTCNN and refine the bounds iof image. Resized to 160,160.
    
    d) Use CV2 to remove background. Similar to step 2b.
    
    e) The resultant image in (d) is input to facenet.
    
    f) The facenet output is input to SVM models (from step 3). Print the SVM face recognition result.
    
    g) The facenet result is input to neural classifier (from step 4). Print the neural classifier result.
    
    
