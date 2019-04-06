# Prepare data for 2D resnet for distances prediction
import numpy as np
import matplotlib.pyplot as plt
# Keras specific
import keras
import keras.backend as K
from keras.regularizers import l2
from keras.losses import mean_squared_error, mean_absolute_error
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Embedding, Dropout, Flatten, UpSampling2D, Input, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
# Model architecture
from elu_resnet_2d_distances import *
from func_utils import *
# Logs-related imports
import sys
import logging
import os

logger.info("First line of the logger - START THE MADNESS")

# START THE MADNESS
# Import the Data Generator from Keras
from distance_generator_data import DataGenerator
# Instantiate DataGenerator - don't keep all the data in memory
# Set args to feed the data generator
params_train = {'paths': TRAINING_PATHS,
                'max_prots': MAX_PROTS,
                'batch_size': BATCH_SIZE,
                'crop_size': CROP_SIZE,
                'pad_size': PAD_SIZE,
                'n_classes': N_CLASSES,
                'class_cuts': CLASS_CUTS,
                'shuffle': False}

# Data generator for training
training_generator = DataGenerator(**params_train)
logger.info("Data already extracted")
#
#
# LOAD THE MODEL AND TRAIN IT
#
#
# Log progress
print("weights", WEIGHTS)
logger.info("weights "+str(WEIGHTS))
# Load model - Using AMSGrad optimizer for speed
model = load_model(GOLDEN_MODEL_PATH, # BASE_MODEL_PATH
        custom_objects={'loss': weighted_categorical_crossentropy(np.array(WEIGHTS)),
                        'softMaxAxis2': softMaxAxis2})
model.compile(optimizer=adam, loss=weighted_categorical_crossentropy(np.array(WEIGHTS)), metrics=["accuracy"]) 
model.summary()
# Log progress
print("Model loaded and ready. Gonna train")
logger.info("Model loaded and ready. Gonna train")

# Train model on dataset
his = model.fit_generator(generator=training_generator,
                          # validation_data=validation_generator,
                          use_multiprocessing=False,
                          workers=1,
                          steps_per_epoch=(MAX_PROTS*BATCH_RATIO)//(BATCH_SIZE),
                          epochs=1,
                          verbose=1,
                          shuffle=True)
# Log progress
logger.info("Training successful "+str(his.history))
model.save(STAGE_MODEL_PATH)
logger.info("Model staged successfully")