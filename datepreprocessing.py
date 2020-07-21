""" download vgg face weights from https://drive.google.com/uc?id=1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo"""
from keras.layers import Dense,Conv2D,Dropout,ZeroPadding2D,MaxPooling2D,Flatten,BatchNormalization,Activation,Convolution2D
from keras.models import Sequential,Model,model_from_json
from keras.preprocessing.image import load_img,img_to_array
import keras.backend as K
import tensorflow as tf
import cv2
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

#Building vggFace model
#Block1
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#Block2
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#Block3
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#Block4
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#Block5
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#Activation block
model.add(Conv2D(4096, (7,7), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))
model.load_weights('vgg_face_weights.h5')
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
print(vgg_face_descriptor.summary())

##Preparing training and validation data
x_train=[]
y_train=[]
person_rep=dict()
person_folders=os.listdir('t')
for i,person in enumerate(person_folders):
    person_rep[i]=person
    image_names=os.listdir('t/'+person+'/')
    for image_name in image_names:
        img=load_img('t/'+person+'/'+image_name,target_size=(224,224))
        img=img_to_array(img)
        img=np.expand_dims(img,axis=0)
        img=preprocess_input(img)
        img_encode=vgg_face_descriptor(img)
        x_train.append(np.squeeze(K.eval(img_encode)).tolist())
        y_train.append(i)
    
    
    x_test=[]
    y_test=[]
    person_folders=os.listdir('v')
    test_image_names=os.listdir('v/'+person+'/')
    for image_name in test_image_names:
        img=load_img('v/'+person+'/'+image_name,target_size=(224,224))
        img=img_to_array(img)
        img=np.expand_dims(img,axis=0)
        img=preprocess_input(img)
        img_encode=vgg_face_descriptor(img)
        x_test.append(np.squeeze(K.eval(img_encode)).tolist())
        y_test.append(i)

#converting train and test data to array
x_train=np.array(x_train) 
y_train=np.array(y_train)
x_test=np.array(x_test) 
y_test=np.array(y_test) 

"""     Saving weighted traing and validation data locally.
        Can be used later for recognition.
"""
with open('x_train_pkl.pkl', 'wb') as file:  
    pickle.dump(x_train, file)
with open('x_test_pkl.pkl', 'wb') as file:  
    pickle.dump(x_test, file)
with open('y_train_pkl.pkl', 'wb') as file:  
    pickle.dump(y_train, file)
with open('y_test_pkl.pkl', 'wb') as file:  
    pickle.dump(y_test, file)