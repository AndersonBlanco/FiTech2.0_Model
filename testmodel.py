# Content in this file is mutable and there is no need to keep it consistent. 
# Serves no importance, testing, creative idea development, problem solving and troubleshooting purposes only 

import tensorflow as tf
from tensorflow import keras 
from keras.callbacks import ModelCheckpoint
import sklearn
import os
import numpy as np
import cv2 
import winsound
from vision import drawSkeleton
import tkinter

#print(os.path.isfile("./punchClassification.keras"))

GRU1 = tf.keras.models.load_model("./ChainedGRU_Arch/punchClassification.keras")#('GRU2.keras')
root = tkinter.Tk()
monitorResolution = (root.winfo_screenheight()+100, root.winfo_screenheight()-200) 

num_videos = 1

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2040)  # set camera as wide as possible
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)



"""val_to_pred = ["good jab", "bad jab - knee level lack", "bad jab - rotation lack",
               "good uppercut", "bad uppercut - rotation lack",
               "good resting", "bad resting", "good straight", "bad straight - lack of defence"]"""



val_pred = ["good jab", "bad jab - knee level lack", 
            "good straight", "bad straight, lack of rotation"," good rest", "bad rest, wrong stance",
            "good kick", "bad kick, don't lounge leg out"]

val_punchClassification_labels = ['jab', 'straightRight', 'upperCut', 'hook', 'rest']

def label_punchClassification(angles):
    pred_y = np.array(GRU1.predict(angles))
    #print("ANGLES:", angles)
    #print("PREDICTION: ", pred_y)
    idx = pred_y[0].argmax(axis = 0)
    p = val_punchClassification_labels[idx]
    print("Prediciton label: ", p)
    print("Raw prediction hot-on eencoding: ", pred_y[0])
    return p


def label(angles):
    pred_y = np.array(GRU1.predict(angles))
    #print("ANGLES:", angles)
    #print("PREDICTION: ", pred_y)
    idx = pred_y[0].argmax(axis = 0)

    return val_pred[idx]


cap = cv2.VideoCapture(0)
counter = 0
a = []
statement = "Put whole body into frame"
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, monitorResolution)
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    try:
        angles, frame = drawSkeleton(frame)
        if counter == 40:
            counter = 0
            numpy_a = np.array(a)
            #print("BOTTOM SECTION: ", numpy_a)
            numpy_a.resize(1,40,8)
            print("numpy_a shape: ", numpy_a.shape)
            statement = label_punchClassification(numpy_a)
            a=[]
        else:
            frame = cv2.circle(frame, (300,300), 40, (0,0,255), -1)
            for i in range(8):
                angles[i] = angles[i]/180
            a.append(angles)
            counter += 1
    except:
        counter = 0
        a=[]
        statement = "Put body in frame"


    cv2.putText(frame, statement, (50,50), cv2.FONT_HERSHEY_DUPLEX, 4, (0,0,0), 4 ,cv2.LINE_AA)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break
    
 
cap.release()
cv2.destroyAllWindows()
