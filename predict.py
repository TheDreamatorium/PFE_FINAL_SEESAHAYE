from keras.models import load_model
import cv2 as cv
from procimg import ProcessImage
import numpy as np

def predict_chars(uploadedImg):
    model = load_model('./model_balanced2.h5')

    image = ProcessImage(uploadedImg)
    characters, points = image.extractCharacters()

    data = dict(zip(points, characters))
    data = dict(sorted(data.items(), key=lambda x: x[0]))
    print(len(characters))

    charDict = dict(zip('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', range(0, 62)))

    print("The predicted text: ")
    output = image.output
    for (key, imgChar) in data.items():
        x, y, w, h = key
        
        prediction = model.predict_classes(imgChar.reshape(1, 28, 28, 1))
        for key, value in charDict.items():
            if value == prediction[0]:
                cv.putText(output,str(key),(x+5,y-5), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(0,255,0),thickness=1)
    return output