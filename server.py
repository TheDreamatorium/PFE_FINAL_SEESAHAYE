from flask import Flask
from flask import jsonify
from flask import request
import numpy as np
import cv2 as cv
import base64
from predict import predict_chars

app = Flask(__name__)

@app.route('/', methods=['POST'])
def sendimage():

    nparr = np.fromstring(request.data, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_ANYCOLOR)

    output = predict_chars(img)

    cv.imwrite("new.png", output, params=None)
    with open("new.png", "rb") as imageFile:
        encoded = base64.b64encode(imageFile.read())

    return str(encoded)
    

if __name__ == "__main__":
    app.run()
