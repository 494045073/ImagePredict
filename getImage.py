import os
import cv2
import json
import random
import numpy as np
from keras.models import load_model
from flask import Flask,request,jsonify

app=Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('getImage.html')

@app.route('/download/',methods=['GET'])
def download():
    folderpath='static/predict/'
    folders=os.listdir(folderpath)
    fileall={}
    for folder in folders:
        fileall[folder]=[]
        folder_file=os.path.join(folderpath,folder)
        if os.listdir(folder_file):
            filesall=os.listdir(folder_file)
            files=random.sample(filesall,20)
            for file in files:
                file_i=os.path.join(folder,file)
                fileall[folder].append({'filename':file,'path':folderpath+file_i})
    return json.dumps(fileall)

if __name__=='__main__':
    app.run()