import os
import json
import cv2
import random
import numpy as np
from keras.models import load_model
from flask import Flask,request,jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('router.html')

def classid(classid):
    return ['apple','banana','orange','mixed'][classid]
def getimage(imagepath):
    img=cv2.resize(cv2.imdecode(np.fromfile(imagepath,dtype=np.uint8),-1),(64,64),interpolation=cv2.INTER_AREA)
    img=img.astype('float32')
    return np.array(img[:,:,:3]/255)

@app.route('/upload', methods=['POST'])
def upload():
    model = load_model('static/resources/model_09-0.23-0.93.hdf5')
    file = request.files['file']
    imagefile = file.filename
    if file:
        dirfile = '../static/resources/received_image'
        if not os.path.isdir(dirfile):
            os.makedirs(dirfile)
        imagefilepath = os.path.join(dirfile, imagefile)
        file.save(imagefilepath)
        inputs = getimage(imagefilepath)[np.newaxis, ...]
        predict_classname = classid(np.argmax(model.predict(inputs)[0]))
        print('对%s预测结果为:%s' % (imagefilepath, predict_classname))
        return jsonify(predict_classname)

@app.route('/download/', methods=['GET'])
def download_file():
    model = load_model('static/resources/model_09-0.23-0.93.hdf5')
    path = './static/predict/'
    folders = os.listdir(path)
    fileList = {}
    for folder in folders:
        fileList[folder] = []
        folder_file = os.path.join(path, folder)
        if os.path.isdir(folder_file):
            fileall = os.listdir(folder_file)
            files=random.sample(fileall,20)
            for file in files:
                imagefile = os.path.join(folder_file, file)
                inputs = getimage(imagefile)[np.newaxis, ...]
                predict_classname = classid(np.argmax(model.predict(inputs)[0]))
                fileList[folder].append({"filename": file, "path": imagefile,'type':predict_classname})
    return json.dumps(fileList)

if __name__ == '__main__':
    app.run('0.0.0.0',port=8020)
