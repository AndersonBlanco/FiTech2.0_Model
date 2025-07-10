import tensorflow as tf 
#import tf_keras as keras 
import sklearn as sklrn 
import keras 
#from keras._tf_keras.keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os 
from sklearn.metrics import accuracy_score, classification_report
from keras.utils import to_categorical
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


def createModel():
    m = keras.Sequential([
    keras.layers.Input(shape=(40,8)),
    keras.layers.GRU(8, return_sequences=False), #units set to 40 due to there being a 40 frame input 
    keras.layers.Dropout(0.1), 
    keras.layers.Dense(3,activation='sigmoid') # was keras.activations.leaky_relu
])
    return m 

punchCalssification_model = createModel()
"""keras.Sequential([
    keras.layers.Input(shape=(40,8)),
    keras.layers.GRU(128, return_sequences=False), #units set to 40 due to there being a 40 frame input 
    keras.layers.Dense(4,activation='softmax')
])"""

#keras.optimizers.SGD(learning_rate=0.01)
punchCalssification_model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), loss =keras.losses.CategoricalCrossentropy(), metrics =['accuracy'])
#print(punchCalssification_model.summary())


"""
flawClassification_model = keras.Sequential([]) 
"""


"""
chained_layers = keras.Sequential([
    keras.layers.Input(shape=(1,40,8)),
    keras.layers.GRU(64, return_sequences=True, activation='relu'),
    keras.layers.GRU(64, return_sequences=False, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(8, activation='softmax')
])

chained_layers.compile(optimizer='adam', loss ='categorical_crossentropy', metrics=['accuracy'])
"""

