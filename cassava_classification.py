import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import  EfficientNetB3
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.regularizers import l2

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report,accuracy_score

# ignoring warnings
import warnings
warnings.simplefilter("ignore")

import os, cv2, json
from PIL import Image

WORK_DIR = "Datasets/cassava-leaf-disease-classification/"

def load_split():
    """
      load the data with file path and split the data
    Args:
        global param: WORK_DIR
    Returns:
        train labels, test labels
    """
    image_labels = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
    train_labels, test_labels = train_test_split(image_labels, train_size = 0.9, shuffle = True,
                                                 random_state = 42)
    return train_labels, test_labels

def image_generate(enable_augmentation:bool,train_labels,TARGET_SIZE,BATCH_SIZE):
    """
      generate the image generator: train generator, valid generator
    Args:
        enable_augmentation: if use augmentation in train generator
        train_labels
        TARGET_SIZE: size of the resized images
        BATCH_SIZE: Batch size of training and generator
    Returns:
        train_generator,validation_generator
    """
    train_labels.label = train_labels.label.astype('str')

    #data augmentation with train images
    if(enable_augmentation):
        train_gen = ImageDataGenerator(validation_split = 0.2,
                                       preprocessing_function = None,
                                       zoom_range = 0.2,
                                       cval = 0,
                                       rotation_range = 60,
                                       horizontal_flip = True,
                                       vertical_flip = True,
                                       fill_mode = 'nearest',
                                       shear_range = 0.15,
                                       height_shift_range = 0.15,
                                       width_shift_range = 0.15) 
    else:
        train_gen = ImageDataGenerator(validation_split = 0.2) 

    train_generator=train_gen.flow_from_dataframe(train_labels,
                                                   directory = os.path.join(WORK_DIR, "train_images"),
                                                   subset = "training",
                                                   x_col = "image_id",
                                                   y_col = "label",
                                                   target_size = (TARGET_SIZE, TARGET_SIZE),
                                                   batch_size = BATCH_SIZE,
                                                   class_mode = "sparse")

    validation_gen= ImageDataGenerator(validation_split = 0.2)

    validation_generator= validation_gen.flow_from_dataframe(train_labels,
                                                             directory = os.path.join(WORK_DIR, "train_images"),
                                                             subset = "validation",
                                                             x_col = "image_id",
                                                             y_col = "label",
                                                             target_size = (TARGET_SIZE, TARGET_SIZE),
                                                             batch_size = BATCH_SIZE,
                                                             class_mode = "sparse")
    return train_generator,validation_generator

def test_generate(test_labels,TARGET_SIZE,BATCH_SIZE):
    """
      generate the test generator
    Args:
        test_labels
        TARGET_SIZE: size of the resized images
        BATCH_SIZE: Batch size of training and generator
    Returns:
        testgenerator
    """

    test_labels.label = test_labels.label.astype('str')

    test_datagen = ImageDataGenerator()

    test_generator = test_datagen.flow_from_dataframe(dataframe=test_labels,
                                                      directory=os.path.join(WORK_DIR, "train_images"),
                                                      x_col="image_id",
                                                      y_col="label",
                                                      target_size=(TARGET_SIZE, TARGET_SIZE),
                                                      batch_size=BATCH_SIZE,
                                                      class_mode='sparse',   #category
                                                      shuffle=False)
    
    return test_generator

def create_CNNmodel(TARGET_SIZE):
    """
      create the custom CNN model
    Args:
        TARGET_SIZE: size of the input images size
    Returns:
        model
    """
    model = models.Sequential([          
            Conv2D(filters=32, kernel_size=(5, 5), padding='Same',activation='relu', input_shape=(TARGET_SIZE, TARGET_SIZE, 3)),
            Conv2D(filters=32, kernel_size=(5, 5), padding='Same',activation='relu'),
            MaxPool2D(pool_size=(2, 2)),

            Conv2D(filters=64, kernel_size=(3, 3), padding='Same',activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), padding='Same',activation='relu'),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Dropout(0.2),

            Conv2D(64, (3, 3), activation='relu',padding="same",kernel_regularizer=l2(0.001)),
            layers.GlobalAveragePooling2D(),


            Flatten(),
            Dense(256, activation="relu"),
            layers.Dropout(0.3),
            Dense(5, activation='softmax')])

    model.compile(optimizer = Adam(lr = 0.001),
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["acc"])
    return model

def create_EfficientNetmodel(TARGET_SIZE):
    """
      create the EfficientNet transfer learning model
    Args:
        TARGET_SIZE: size of the input images size
    Returns:
        model
    """
    model = models.Sequential([          
            EfficientNetB3(include_top=False, weights='imagenet',
                            input_shape = (TARGET_SIZE, TARGET_SIZE, 3)),
            layers.GlobalAveragePooling2D(),
            layers.Flatten(),
            Dense(256, activation="relu"),
            layers.Dropout(0.3),
            Dense(5, activation='softmax')])

    model.compile(optimizer = Adam(lr = 0.001),
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["acc"])
    return model

def model_training(model,model_name, train_generator,validation_generator,
                   EPOCHS, STEPS_PER_EPOCH,VALIDATION_STEPS):
    """
      train the CNN model
    Args:
        model
        model_name: model saved and checkpoint name
        train generator, validation generator
        Epochs,STEPS_PER_EPOCH,VALIDATION_STEPS
    Returns:
        trained model
    """
    
    model_save = ModelCheckpoint(os.path.join("Results", "best" + model_name + ".h5"), 
                             save_best_only = True, 
                             save_weights_only = True,
                             monitor = 'val_loss', 
                             mode = 'min', verbose = 1)
    
    early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.0001, 
                               patience = 4, mode = 'min', verbose = 1,
                               restore_best_weights = True)
    
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
                                  patience = 2, min_delta = 0.001, 
                                  mode = 'min', verbose = 1)


    history = model.fit_generator(
        train_generator,
        steps_per_epoch = STEPS_PER_EPOCH,
        epochs = EPOCHS,
        validation_data = validation_generator,
        validation_steps = VALIDATION_STEPS,
        callbacks = [model_save, early_stop, reduce_lr])
    
    model.save(os.path.join("Results", model_name + ".h5"))

    return model

def compute_classweight(train_labels):
    """
      compute the class weight from train labels
      range from 0 to 1
    Args:
        train labels
    Returns:
        class weight dict
    """
    current_balance = train_labels['label'].value_counts(normalize=True)
    class_weight = {0: (1 - current_balance['0']) / (1 - current_balance.min()),
                    1: (1 - current_balance['1']) / (1 - current_balance.min()),
                    2: (1 - current_balance['2']) / (1 - current_balance.min()),
                    3: (1 - current_balance['3']) / (1 - current_balance.min()),
                    4: (1 - current_balance['4']) / (1 - current_balance.min())}
    return class_weight

def weighted_training(model,model_name, train_generator,validation_generator,
                   EPOCHS, STEPS_PER_EPOCH,VALIDATION_STEPS,class_weight):  
    """
      train the CNN model with class_weight
    Args:
        model
        model_name: model saved and checkpoint name
        train generator, validation generator
        Epochs,STEPS_PER_EPOCH,VALIDATION_STEPS
        class_weight: value counts of labels
    Returns:
        trained model
    """
    
    model_save = ModelCheckpoint(os.path.join("Results", "best" + model_name + ".h5"), 
                             save_best_only = True, 
                             save_weights_only = True,
                             monitor = 'val_loss', 
                             mode = 'min', verbose = 1)
    
    early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.0001, 
                               patience = 4, mode = 'min', verbose = 1,
                               restore_best_weights = True)
    
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
                                  patience = 2, min_delta = 0.001, 
                                  mode = 'min', verbose = 1)


    history = model.fit_generator(
        train_generator,
        steps_per_epoch = STEPS_PER_EPOCH,
        epochs = EPOCHS,
        validation_data = validation_generator,
        validation_steps = VALIDATION_STEPS,
        class_weight=class_weight,
        callbacks = [model_save, early_stop, reduce_lr])
    
    model.save(os.path.join("Results", model_name + ".h5"))

    return history,model

def test(model, test_generator):
    """
      Evaluatethe model
    Args:
        model
        test generator
    Results:
        Accuracy, F1 score, recall, precision(weighted)
        classification report
    """
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    true_classes = test_generator.classes


    accuracy = accuracy_score(true_classes , y_pred_classes)

    f1 = f1_score(true_classes, y_pred_classes, average='weighted')

    recall = recall_score(true_classes, y_pred_classes, average='weighted')

    precision = precision_score(true_classes, y_pred_classes,average="weighted")

    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:",precision)
    print("F1 Score:", f1)
    print(classification_report(true_classes,  y_pred_classes))