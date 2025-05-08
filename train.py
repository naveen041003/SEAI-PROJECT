# Importing other necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, GaussianNoise
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.optimizers import SGD, Adam, schedules
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow import keras

from datetime import datetime
import os

##############################################################################################
"""
Remarks: 
1. Put this train script outside a train folder, inside train folder there should be test_data, train_data.
         
2. Specify the model_name. The weights and logs folder will be saved inside trainfolder->model_name

3. Select GPU, 0 or 1
         
"""
# Specify the paths
##############################################################################################
model_name = '1.5kClass_32x32'

#get current file path
current = os.path.dirname(os.path.realpath(__file__))

#get current file path
train_dataset_folder = r"C:\Users\andre\Simple-Food-Image-Classifier\Dataset1.5kClass"
train_folder_dir = os.path.join(current, train_dataset_folder)

weights_dir      = os.path.join(train_folder_dir ,model_name,'weights')
logs_dir         = os.path.join(train_folder_dir ,model_name,'logs')
train_data_dir   = os.path.join(train_folder_dir ,'train_data')
test_data_dir    = os.path.join(train_folder_dir ,'test_data')

# Select GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Specify hyperparameters
num_classes = len(os.listdir(train_data_dir))
EPOCH = 200
Batch_SIZE = 16
height = 32
width = 32

# Set Piecewise Constant Decay learning schedule
boundaries = [10]
values = [0.001,0.0001]
lr_schedule = schedules.PiecewiseConstantDecay(boundaries, values)

# Creating weights and logs folder
if not os.path.exists(weights_dir):
    print("Weights dir not found... creating a new one")
    os.makedirs(weights_dir)
if not os.path.exists(logs_dir):
    print("Logs dir not found... creating a new one")
    os.makedirs(logs_dir)

# Data preparation
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   brightness_range=[0.9,1.1],
                                   rotation_range = 180,
                                   shear_range=0.2,
                                   zoom_range = 0.2,
                                   width_shift_range=0.25,
                                   height_shift_range=0.25,
                                   horizontal_flip = True,
                                   vertical_flip   = True,
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_data_dir,
                                                 target_size = (height, width),
                                                 batch_size = Batch_SIZE,
                                                 shuffle=True,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_data_dir,
                                            target_size = (height, width),
                                            batch_size = Batch_SIZE,
                                            shuffle=False,
                                            class_mode = 'categorical')

print(training_set.class_indices)
##############################################################################################

def model_definition(height,width,num_classes,channel):
        model = tf.keras.models.Sequential([
            #Convolution Block
            tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', input_shape = (height, width, channel), padding='same'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', padding='same'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', padding='same'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', padding='same'),
            tf.keras.layers.MaxPooling2D(2, 2),

            #Flatten
            tf.keras.layers.Flatten(),
            
            #Fully Connected Layer
            tf.keras.layers.Dense(256, activation = 'relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation = 'relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation = 'relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])


        return model


# Start training
model = model_definition(height,width,num_classes,channel=3)


print(model.summary())
model.compile(optimizer=Adam(learning_rate=lr_schedule),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Setting callbacks parameters
modelcheckpoint_callback = ModelCheckpoint(filepath = os.path.join(weights_dir,"weights.{epoch:02d}-{val_accuracy:.4f}-{val_loss:.4f}.h5"), 
                                           monitor = 'val_accuracy', # comment out if saving every epoch
                                           mode = 'max',             # comment out if saving every epoch
                                           save_best_only = True
                                           )
                                                                   
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1)

csv_logger = CSVLogger(r'training.log')

history = model.fit(training_set,
          epochs = EPOCH,
          validation_data = test_set, callbacks = [modelcheckpoint_callback,tensorboard_callback, csv_logger])




# Stopped training
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
