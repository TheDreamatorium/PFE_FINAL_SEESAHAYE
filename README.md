# PFE_FINAL_SEESAHAYE

Descriptions:
API in python to detect handwritten characters on paper using OpenCV and the EMNIST dataset.
When the base64 encoded image is received, it is converted back to a PNG image. Then the detection is done by the python script.

Instructions:
--Install Virtual Environment
pip3 install virtualenv

--Create the virtual environment
virtualenv mypython

--Activate the virtual environment
source mypython/bin/activate

--install packages
pip3 install requirements.txt

--Start the server (commands)
1. export FLASK_APP=server.py
2. flask run --host=0.0.0.0 --without-threads
