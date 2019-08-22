print("loading preprocessing modules")
import cv2
import json
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
print("loaded preprocessing modules")

class PreProcess():

    def __init__(self):
        print("loading vgg16")
        self.session=tf.Session()
        self.graph=tf.get_default_graph()
        with self.graph.as_default():
            with self.session.as_default():
                image_model=VGG16(include_top=True, weights='imagenet')
                transfer_layer=image_model.get_layer('fc2')
                self.image_model_transfer=Model(inputs=image_model.input,
                                           outputs=transfer_layer.output)
                # self.image_model_transfer._make_predict_function()
        print("loaded vgg16")


    def extract_features(self,train_img,batch_size,frames=80):
        features=[]
        for x in range(0,frames,batch_size):
            train_img1=np.array(train_img[x:x+batch_size])
            train_img1=preprocess_input(train_img1)
            with self.graph.as_default():
                with self.session.as_default():
                    features.append(self.image_model_transfer.predict(train_img1))
        print("appending features")

        #convert the features to numpy array
        print("converting features to numpy array")
        features=np.asarray(features)
        #reshape features from (8,10,4096) to (80,4096)
        print("reshaping features")
        features=features.reshape(-1,4096)
        #convert them back to list as json doesnt accept numpy arrays
        features=features.tolist()

        return features
