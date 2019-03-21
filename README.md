# PFE_FINAL_SEESAHAYE

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