#model development
#model will be constructed and its weights and biases will be saved from here as well

import tensorflow as tf 
#import tf_keras as keras 
import sklearn as sklrn 
import keras 
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os 
from sklearn.metrics import accuracy_score, classification_report
from keras.utils import to_categorical
import coremltools as crml

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from chainedLayers_model import punchCalssification_model, createModel
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./punchClassification_checkpoint.ckpt",save_weights_only=True, verbose=1)

def loadModel():
    m = createModel()
    m.compile(optimizer=keras.optimizers.SGD(learning_rate=1.0), loss =keras.losses.CategoricalCrossentropy(), metrics =['accuracy'])
    m.load_weights("./punchClassification.weights.h5")
    return m

"""
Data classification: 
Punch types: 
- Jab
- StrightRight 
- Upper Cut
- Hook 

punch classification label architecture: [jab, straightRight, rest]
"""
def getX_getY(path, label):
    xp = []
    y = []
    counter = 0

    for vid_folders in os.listdir(path):
        #print(vid_folders)
        counter +=1
        y.append(label)

        data = np.load(path + "/" + vid_folders)
       # print(data)

        x  = []
        for i in range(40):
            temp = []
            for j in range(8):
                temp.append(data[i][j]/180)
            x.append(temp)
        
        xp.append(x)

        """
        for angles in os.listdir(path + '/' + vid_folders):
            data = np.load( (path + '/' + vid_folders +'/' + str(angles)) )

            #y stores the label of the video
            #x is a 40 files containing 40 angles of a persons movement
            #we could try to optimize further by having min val and max val instead of 0 to 180 angles
            temp = []
            for i in range(8):
                temp.append(data[i]/180)
            x.append(np.array(temp))
        """
            
    xp= np.array(xp)
    yp = np.array(y)

    
    xp.resize(counter,40, 8)
    #print("X SHAPE:", xp.shape)
    return xp, yp

def getX_getY_2(path, label):
    x = []
    y = []
    count = 0 

 
    for video_folder in os.listdir(path):
        for vidFolder in os.listdir(path):
            y.append(label)
            #print(vidFolder)
            p = path + f"/{vidFolder}"
            try:
                 tmp = []
                 for npyFile in os.listdir(p):
                    _p = p + "/" + npyFile
                    #print(_p)
                    data = np.load(_p)
                    tmp.append(data)
                 x.append(tmp)
                    
            except Exception as e:
                print(e, "at: ", p) 
                continue

    return np.array(x), np.array(y)

def convertModel(model, name):
    try:
        m = crml.convert(model, source = "tensorflow")
        m.save(name)
        print('model saved!!')
        return m
    except Exception as e:
        print(e)
        return e 

def fit_and_test_model_with_loadedWeights(x_train,y_train, x_test, y_test):
    loadedModel = loadModel()
    loadedModel.load_weights("./punchClassification.weights.h5")
    
    print("fitting..")
    loadedModel.fit(x_train,y_train, epochs=25, batch_size=2) 
    print("fitting complete..")

    print("testing.....")
    loadedModel.evaluate(x_test, y_test, verbose='auto')
    print('testing complete...')


    print('saving model....')
    loadedModel.save("./punchClassification.keras")
    loadedModel.save_weights("./punchClassification.weights.h5")
    print('model saved....')

    print('converting to coreml model file..')
    crmlModel = convertModel(loadedModel, "./punchClassification_coreml2.mlpackage")
    print('coreml model file converted')

def fit_and_test_bare_model(x_train,y_train, x_test, y_test, x_train2, y_train2, x_test2, y_test2):
    print("fitting 1..")
    punchCalssification_model.fit(x_train,y_train, epochs=25, batch_size=2) 
    print("fitting 1 complete..")
    print("testing 1..")
    punchCalssification_model.evaluate(x_test, y_test, verbose='auto')
    print('testing 1 complete...')

    print("fitting 2..")
    punchCalssification_model.fit(x_train2,y_train2, epochs=25, batch_size=2) 
    print("fitting 2 complete..")
    print("testing 2..")
    punchCalssification_model.evaluate(x_test2, y_test2, verbose='auto')
    print('testing 2 complete...')


    print('saving model....')
    punchCalssification_model.save("./punchClassification.keras")
    punchCalssification_model.save_weights("./punchClassification.weights.h5")
    print('model saved....')


    print('converting to coreml model file..')
    crmlModel = convertModel(punchCalssification_model, "./punchClassification_coreml4.mlpackage")
    print('coreml model file converted')

#converting model directly from directory:
def convertModelFromPath():
    m = createModel()
    m.compile(optimizer=keras.optimizers.SGD(learning_rate=1.0), loss =keras.losses.CategoricalCrossentropy(), metrics =['accuracy'])
    m.load_weights("./punchClassification.weights.h5")
    convertModel(model, "./punchClassification_coreml3.mlpackage")


####################### Punch Classification Modoel Data Processing --- START----###########################################################
x_jabs_good, y_jabs_good = getX_getY('../newData/jab/good', [1, 0,0])
x_jabs_bad, y_jabs_bad = getX_getY('../newData/jab/bad', [1, 0,0])

x_straightRight_good, y_straightRight_good = getX_getY('../newData/straightRight/good', [0, 1,0])
x_straightRight_bad, y_straightRight_bad = getX_getY('../newData/straightRight/bad',[0, 1, 0])

x_rest_good, y_rest_good = getX_getY('../newData/rest/good', [0,0,1])
x_rest_bad, y_rest_bad = getX_getY('../newData/rest/bad', [0,0,1])

#x_jabs2_good, y_jabs2_good = getX_getY_2('../Data/past/jab/good/angles', label=[1, 0,0,0, 0]) #NOTE: 1 copied f_30.npy and psted it 9 times to avoid inhomonogous numbers of frames sets of angles
#x_jabs_bad_kneeLvlLack, y_jabs_bad_kneeLvlLack = getX_getY_2('../Data/past/jab/bad/angles/knee_lvl_lack', label=[1, 0,0,0, 0])
#x_jabs_bad_rotationLack, y_jabs_bad_rotationLack = getX_getY_2('../Data/past/jab/bad/angles/rotation_lack', label=[1, 0,0,0, 0])

x_straightRight2_good, y_straightRight2_good = getX_getY_2('../Data/past/straight_right/good/angles',[0, 1,0])
x_straightRight2_bad, y_straightRight2_bad = getX_getY_2('../Data/past/straight_right/bad/angles/straight_defence_lack', [0, 1,0])

x_rest2_good, y_rest2_good = getX_getY_2("../Data/past/rest/good/angles",[0,0,1])
x_rest2_bad, y_rest2_bad = getX_getY_2('../Data/past/rest/bad/angles',[0,0,1])

print("x rest2 good shape: ", x_rest2_good.shape)
print("y rest2 good shape: ", y_rest2_good.shape)

print("x straightRight2 good shape: ", x_straightRight2_good.shape)
print("y straightRight2 good shape: ", y_straightRight2_good.shape)
#x_upperCut_good, y_upperCut_good = getX_getY('../Data/past/uppercut/good/angles', label=[0, 0,1,0,0])
#x_upperCut_bad, y_upperCut_bad = getX_getY('../Data/past/uppercut/bad/angles/upper_knee_lvl_lack', label=[0, 0,1,0,0])

print("X jab good shape: ", x_jabs_good.shape)
print("Y jab good shape: ", y_jabs_good.shape)

print("X jab bad shape: ", x_jabs_bad.shape)
print("Y jab bad shape: ", y_jabs_bad.shape)

print("X straightRight good shape: ", x_straightRight_good.shape)
print("Y straightRight good shape: ", y_straightRight_good.shape)

print("X straightRight bad shape: ", x_straightRight_bad.shape)
print("Y straightRight bad shape: ", y_straightRight_bad.shape)
    

x_jabs = np.add(x_jabs_good, x_jabs_bad)
y_jabs = np.add(y_jabs_good, y_jabs_bad)

x_jabs_train, y_jabs_train, x_jabs_test, y_jabs_test = train_test_split(x_jabs, y_jabs, test_size=0.2, random_state=1)
print(x_jabs.shape, y_jabs.shape)

#jabs_x_set = np.concatenate((x_jabs_good, x_jabs_bad), axis=2)
#jabs_y_set = np.concatenate((y_jabs_good, y_jabs_bad), axis = 2)

#straightRight_x_set = np.concatenate((x_straightRight_good, x_straightRight_bad), axis =2)
#straightRight_x_set = np.concatenate((y_straightRight_good, y_straightRight_bad), axis =2)

trainingSet_x = np.concatenate((x_jabs_good, x_straightRight_good,  x_rest_good), axis = 0) #x_straightRight2_good, x_straightRight2_bad, x_rest2_good, x_rest2_bad), axis = 0)
trainingSet_y = np.concatenate((y_jabs_good, y_straightRight_good,  y_rest_good), axis = 0)# y_straightRight2_good, y_straightRight2_bad, y_rest2_good, y_rest2_bad), axis = 0)

testSet_x = np.concatenate((x_jabs_good, x_straightRight_good,  x_rest_good,), axis=0)  #x_straightRight2_good, x_straightRight2_bad, x_rest2_good, x_rest2_bad), axis = 0)
testSet_y = np.concatenate((y_jabs_good, y_straightRight_good,  y_rest_good), axis = 0) #y_straightRight2_good, y_straightRight2_bad, y_rest2_good, y_rest2_bad), axis = 0)


trainingSet_x2 = np.concatenate((x_jabs_bad, x_straightRight_bad,  x_rest_bad), axis = 0) #x_straightRight2_good, x_straightRight2_bad, x_rest2_good, x_rest2_bad), axis = 0)
trainingSet_y2 = np.concatenate((y_jabs_bad, y_straightRight_bad,  y_rest_bad), axis = 0)# y_straightRight2_good, y_straightRight2_bad, y_rest2_good, y_rest2_bad), axis = 0)

testSet_x2 = np.concatenate((x_straightRight_bad,  x_rest_bad), axis=0)  #x_straightRight2_good, x_straightRight2_bad, x_rest2_good, x_rest2_bad), axis = 0)
testSet_y2 = np.concatenate((y_straightRight_bad,  y_rest_bad), axis = 0) #y_straightRight2_good, y_straightRight2_bad, y_rest2_good, y_rest2_bad), axis = 0)

print(trainingSet_x.shape,trainingSet_y.shape)
print(testSet_x.shape,testSet_y.shape)


####################### Punch Classification Modoel Data Processing ----- END----###########################################################



####################### Flaw CLassification Modoel Data Processing + model architecture --- START----###########################################################

class CustomLayer():
    def __init__(self, init_inputShape, **others):
        super().__init__(**others)
        self.init_inputShape = init_inputShape
        self.init_input = None

        self.gru1=tf.keras.layers.GRU(8, return_sequences=True)
        self.Dropout1=tf.keras.layers.Dropout(0.1)
        self.Dense1 = tf.keras.layers.Dense(3, activation='sigmoid')

        self.gru2=tf.keras.layers.GRU(8, return_sequences=False)
        self.Dropout2=tf.keras.layers.Dropout(0.1)
        self.Dense2 = tf.keras.layers.Dense(6, activation='sigmoid')

        self.gru1_output = None
        self.dense1_output = None
        self.gru2_output = None
        self.dense2_output = None


        self.conctainate = tf.keras.layers.Concatenate(axis = 2)
        self.repeat_vector = tf.keras.layers.RepeatVector(self.init_inputShape[0])

    def predict_with_gru1(self):
      gru1_output = self.gru1(self.init_input)
      dropout1_output = self.Dropout1(gru1_output, training = True)
      dense1_output = self.Dense1(dropout1_output)
      self.gru1_output = gru1_output
      self.dense1_output = dense1_output
      return dense1_output

    def predict_with_gru2(self, dense1_output):
      input_repeated = self.repeat_vector(dense1_output[:, -1, :])
      combinedInput = self.conctainate([input_repeated, self.init_input])
      gru2_output = self.gru2(combinedInput)
      dropout2_output = self.Dropout2(gru2_output, training=True)
      dense2_output = self.Dense2(dropout2_output)
      self.gru2_output = gru2_output
      self.dense2_output = dense2_output
      return dense2_output


    def run_layer(self, init_input):
      self.init_input = init_input
      gru1 = self.predict_with_gru1()
      gru2 =self.predict_with_gru2(gru1)

      return gru2


customLayer = CustomLayer((40,8))
mainInput = keras.layers.Input(shape=(40,8))
lastGRUOutput = customLayer.run_layer(mainInput)
CHAINED_MODEL = keras.Model(inputs = mainInput, outputs=[lastGRUOutput])
CHAINED_MODEL.compile(optimizer=keras.optimizers.SGD(learning_rate=0.5), loss =keras.losses.CategoricalCrossentropy(), metrics =['accuracy'])
#print(CHAINED_MODEL.summary())

"""
prediciton return label architecture: [jab_lack_of_rotation, jab_correct, straight_right_lack_of_rotation, straight_right_correct, rest_bad_stance, rest_correct ]
"""
cx_jabs_good, cy_jabs_good = getX_getY('../newData/jab/good', [0, 1, 0,0, 0,0])
cx_jabs_bad, cy_jabs_bad = getX_getY('../newData/jab/bad', [1, 0, 0,0, 0,0])

cx_straightRight_good, cy_straightRight_good = getX_getY('../newData/straightRight/good', [0,0,0, 1,0, 0])
cx_straightRight_bad, cy_straightRight_bad = getX_getY('../newData/straightRight/bad',[0, 0, 1, 0, 0, 0])

cx_rest_good, cy_rest_good = getX_getY('../newData/rest/good', [0,0,0,0,0,1])
cx_rest_bad, cy_rest_bad = getX_getY('../newData/rest/bad', [0,0,0,0,1,0])

CHAINED_MODEL_x_traininng_set = np.concatenate((cx_jabs_good, cx_jabs_bad, cx_straightRight_good, cx_straightRight_bad, cx_rest_good, cx_rest_bad), axis = 0)
CHAINED_MODEL_y_traininng_set = np.concatenate((cy_jabs_good, cy_jabs_bad, cy_straightRight_good, cy_straightRight_bad, cy_rest_good, cy_rest_bad), axis = 0)

CHAINED_MODEL_x_testing_set = np.concatenate((cx_jabs_good, cx_jabs_bad, cx_straightRight_good, cx_straightRight_bad, cx_rest_good, cx_rest_bad), axis = 0)
CHAINED_MODEL_y_testing_set = np.concatenate((cy_jabs_good, cy_jabs_bad, cy_straightRight_good, cy_straightRight_bad, cy_rest_good, cy_rest_bad), axis = 0)

print("CHAINED_MODEL data shapes: ")
print("X_TEST: ", CHAINED_MODEL_x_testing_set.shape)
print("Y_TEST: ", CHAINED_MODEL_y_testing_set.shape)
print("X_TRAIN: ", CHAINED_MODEL_x_traininng_set.shape)
print("Y_TRAIN: ", CHAINED_MODEL_y_traininng_set.shape)
####################### Flaw CLassification Modoel Data Processing + model architecture  --- END----###########################################################





#######################Model fitting (both Punhc Classification and Flaw CLassification) section below ***no data manipulation ###########################################################

#fit_and_test_bare_model(trainingSet_x, trainingSet_y, testSet_x, testSet_y, trainingSet_x2, trainingSet_y2, testSet_x2, testSet_y2)
#fit_and_test_model_with_loadedWeights(x_jabs_good, y_jabs_good, x_jabs_bad, y_jabs_bad)
#fit_and_test_model_with_loadedWeights(x_rest_good, y_rest_good, x_rest_bad, y_rest_bad)

print("Training CHAINED_MODEL..")
CHAINED_MODEL.fit(CHAINED_MODEL_x_traininng_set, CHAINED_MODEL_y_traininng_set, epochs = 25, batch_size=0)
print("Training CHAINED_MODEL complete..")

print("Testing CHAINED_MODEL..")
CHAINED_MODEL.evaluate(CHAINED_MODEL_x_testing_set, CHAINED_MODEL_y_testing_set, verbose='auto')
print("Testing CHAINED_MODEL complete..")

print("saving model..")
CHAINED_MODEL.save("CHAINED_MODEL.keras")
punchCalssification_model.save_weights("./CHAINED_MODEL.weights.h5")
print('model saved....')

print('converting to coreml model file..')
crmlModel = convertModel(CHAINED_MODEL, "./CHAINED_MODEL_coreml.mlpackage")
print('coreml model file converted')
