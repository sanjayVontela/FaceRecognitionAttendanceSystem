from keras.layers import Dense,Conv2D,Dropout,ZeroPadding2D,MaxPooling2D,Flatten,BatchNormalization,Activation,Convolution2D
from keras.models import Sequential,Model,model_from_json
from keras.preprocessing.image import load_img,img_to_array
import keras.backend as K
import tensorflow as tf
import cv2
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import tkinter
import tkinter.font
import cv2
import PIL.Image, PIL.ImageTk
import time
import datetime
from database_psql import Database
class App:
    def __init__(self, window, window_title, video_source=0):
         
        self.window = window
        self.window.geometry('800x650')
        self.window.title(window_title)
        self.window.configure(bg='gray20')
        self.video_source = video_source
        


         # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height, bg = "gray40")
        self.canvas.pack(anchor = tkinter.CENTER, padx = 50, pady = 100)
        

         # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
        
        self.window.mainloop()

    

    def update(self):
         # Get a frame from the video source
        ret, frame,name = self.vid.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)
        # return name

class MyVideoCapture:
    def __init__(self, video_source=0):
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
        model.add(Conv2D(64,(3,3),activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(64,(3,3),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))


        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128,(3,3),activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128,(3,3),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256,(3,3),activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256,(3,3),activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256,(3,3),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512,(3,3),activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512,(3,3),activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512,(3,3),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))


        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512,(3,3),activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512,(3,3),activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512,(3,3),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        model.add(Conv2D(4096, (7,7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))
        model.load_weights(r'D:\projects\vggface\vgg\vgg_face_weights.h5')
        self.vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
        print(model.summary())
        with open(r'D:\projects\vggface\vgg\x_train_pkl.pkl', 'rb') as file:  
            x_train = pickle.load(file)
        with open(r'D:\projects\vggface\vgg\x_test_pkl.pkl', 'rb') as file:  
            x_test = pickle.load(file)
        with open(r'D:\projects\vggface\vgg\y_train_pkl.pkl', 'rb') as file:  
            y_train = pickle.load(file)
        with open(r'D:\projects\vggface\vgg\y_test_pkl.pkl', 'rb') as file:  
            y_test = pickle.load(file)
        self.classifier_model=Sequential()
        self.classifier_model.add(Dense(units=100,input_dim=x_train.shape[1],kernel_initializer='glorot_uniform'))
        self.classifier_model.add(BatchNormalization())
        self.classifier_model.add(Activation('tanh'))
        self.classifier_model.add(Dropout(0.3))
        self.classifier_model.add(Dense(units=10,kernel_initializer='glorot_uniform'))
        self.classifier_model.add(BatchNormalization())
        self.classifier_model.add(Activation('tanh'))
        self.classifier_model.add(Dropout(0.2))
        self.classifier_model.add(Dense(units=6,kernel_initializer='he_uniform'))
        self.classifier_model.add(Activation('softmax'))
        self.classifier_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer='nadam',metrics=['accuracy'])
        self.classifier_model.fit(x_train,y_train,epochs=5,batch_size=16,validation_data=(x_test,y_test))
    
         # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

         # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    def predected_name(self,detected_face):
        person_rep={0:'amar',1:'anirudh',2:'manish',3:'sanjay'}
        crop_img=img_to_array(detected_face)
        crop_img=np.expand_dims(crop_img,axis=0)
        crop_img=preprocess_input(crop_img)
#         print(crop_img.shape)
        img_encode=self.vgg_face_descriptor(crop_img)
#         print(img_encode.shape)
      # Make Predictions
        embed=K.eval(img_encode)
#         print(embed.shape)
        person=self.classifier_model.predict(embed)
#         print(person)
#         print(np.argmax(person))
        
        name=person_rep[np.argmax(person)]
        time_now = str(datetime.datetime.now().time())
        # print(time_now)
        # time.sleep(5)
        database.attendance(time_now,name)
        return name
        #print(name)

    
    def get_frame(self):
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            name=""
            if ret:
                faces = face_classifier.detectMultiScale(frame, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
                    detected_face = frame[int(y):int(y+h), int(x):int(x+w)]
                    detected_face = cv2.resize(detected_face, (224, 224))    
                    name = self.predected_name(detected_face)
                    cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
                    print(name)
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),name)
            else:
                return (ret, None)
        else:
            return (ret, None)
    
    
     # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

 # Create a window and pass it to the Application object
database = Database()
database.update()
App(tkinter.Tk(), "Attendance - Face recognition")

