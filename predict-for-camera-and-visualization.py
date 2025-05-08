#-----------------------------------------------------------------#
#------------------------IMPORTING MODULES------------------------#
#-----------------------------------------------------------------#
from keras.preprocessing import image
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

'''
ABOUT:

This script requires a webcam and allows a real time detection of model and shows the features extracted in real time inside the activation layer.
This is purely for visual purposes and to visualize the features extracted by the image for testing and evaluation of results.

It is not to be used for benchmarking, but to just see and explain the extraction of features for the model and how it is detecting.
It allows an outsider other than the developers to see and learn about the inner workings of the model.
'''

#get current file path
current = os.path.dirname(os.path.realpath(__file__))


# Opens the inbuilt camera of laptop to capture video.
cap = cv2.VideoCapture(cv2.CAP_DSHOW)
font = cv2.FONT_HERSHEY_SIMPLEX

#state camera labels
class_labels = {0:'Ikura Sushi' , 1:'Mushroom', 2:'Onion', 3:'Others'}
class_colors = {0:(0, 0, 0), 1:(255, 0, 0), 2:(0, 255, 0), 3:(0, 0, 255)}

#get weights path
WEIGHTS_PATH = r""

model = load_model(os.path.join(current, WEIGHTS_PATH))

#Probablity model showing percentage of each class
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

#activate image model to show the features extracted from activation layer
layer_outputs = [layer.output for layer in model.layers[:12]] 
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

#image dimension to resize
img_width = 32
img_height = 32
dim = (int(img_width), int(img_height))

#get visual feedback when camera is open
while(cap.isOpened()):
    ret, frame = cap.read()
    
    #process camera frame input before sending to model
    frame = cv2.flip(frame, 1)
    x, y, z = frame.shape
    frame = frame[100:y-100, 50:x-50]

    test_image = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    #send image to model for real time prediction
    predictions = probability_model.predict(test_image)

    #write prediction results on image
    cv2.putText(frame, class_labels[np.argmax(predictions)], (10,30), font, 1, class_colors[np.argmax(predictions)], 2, cv2.LINE_AA)
    cv2.putText(frame, str(predictions[0][0]), (10,300), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, str(predictions[0][1]), (10,320), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, str(predictions[0][2]), (10,340), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, str(predictions[0][3]), (10,360), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    
    # Extracts the outputs of the top 12 layers
    activations = activation_model.predict(test_image) # Returns a list of five Numpy arrays: one array per layer activation

    #save latest frame incase camera stops
    try:
        plt.imsave(os.path.join(current, "layer1.jpg"), activations[0][0, :, :, 4], cmap='viridis')
        plt.imsave(os.path.join(current, "layer2.jpg"), activations[1][0, :, :, 4], cmap='viridis')
    except:
        continue
    
    #resize from 32x32 to show image clearly.
    featureframe1 = cv2.resize(cv2.imread(os.path.join(current, "layer1.jpg")), (int(frame.shape[0]/2), int(frame.shape[1]/2)))
    featureframe2 = cv2.resize(cv2.imread(os.path.join(current, "layer2.jpg")), (int(frame.shape[0]/2), int(frame.shape[1]/2)))

    #concatenate image to show in window.
    im_v = cv2.vconcat([
        featureframe1,
        featureframe2,
    ])

    im_h = cv2.hconcat([
        im_v,
        frame,
    ])
    

    if ret == False:
        break
    
    #show real time update of image and features
    cv2.imshow("Cam Classifier", im_h)
    keypressed = cv2.waitKey(30)
    if keypressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#deletes saved image after running script.
os.remove(os.path.join(current, "layer1.jpg"))
os.remove(os.path.join(current, "layer2.jpg"))