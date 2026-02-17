import cv2
import os
from tensorflow.keras.models import load_model, Sequential, model_from_json
import numpy as np
from pygame import mixer
import time
import h5py
import json

def load_compat_model(filepath):
    try:
        # Try standard loading first
        return load_model(filepath)
    except (TypeError, ValueError) as e:
        print(f"Standard load failed, attempting legacy patch: {e}")
        # Legacy patch for Keras 3+ loading Keras 2.x H5 models
        with h5py.File(filepath, 'r') as f:
            model_config = f.attrs.get('model_config')
            if model_config is None:
                raise ValueError('No model found in config file.')
            if isinstance(model_config, bytes):
                model_config = model_config.decode('utf-8')
            model_config_dict = json.loads(model_config)

        if isinstance(model_config_dict['config'], list):
            print("Patching legacy config: converting list to dict...")
            layers_config = model_config_dict['config']
            # Patch first layer input shape
            if len(layers_config) > 0:
                first_layer_config = layers_config[0]['config']
                if 'batch_input_shape' in first_layer_config:
                    batch_shape = first_layer_config['batch_input_shape']
                    if len(batch_shape) == 4 and batch_shape[0] is None:
                        first_layer_config['input_shape'] = batch_shape[1:]
                        first_layer_config.pop('batch_input_shape', None)
            
            new_config = {'layers': layers_config, 'name': 'sequential'}
            try:
                model = Sequential.from_config(new_config)
            except Exception as e_conf:
                print(f"Sequential.from_config failed: {e_conf}")
                # Fallback to model_from_json if strictly needed, but Sequential.from_config is cleaner
                model_config_dict['class_name'] = 'Sequential'
                model_config_dict['config'] = new_config
                model = model_from_json(json.dumps(model_config_dict))
                
            model.load_weights(filepath)
            return model
        else:
            raise e

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier(r'haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(r'haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(r'haar cascade files\haarcascade_righteye_2splits.xml')



lbl=['Close','Open']

model = load_compat_model('models/cnnCat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        #rpred = model.predict_classes(r_eye)
        # rpred = model.predict(r_eye)
        # rpred = np.round(rpred).astype(int)
        rpred = np.argmax(model.predict(r_eye), axis=-1)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        #lpred = model.predict_classes(l_eye)
        # lpred = model.predict(l_eye)
        # lpred = np.round(lpred).astype(int)
        lpred = np.argmax(model.predict(l_eye), axis=-1)
        print(lpred)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>12):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
