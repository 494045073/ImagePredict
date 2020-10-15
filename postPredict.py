import os
import cv2
import numpy as np
from keras.models import load_model
from flask import Flask,request,jsonify

app=Flask(__name__)

def classid(classid):
    return ['apple','banana','orange','mixed'][classid]
def getimage(imagepath):
    img=cv2.resize(cv2.imdecode(np.fromfile(imagepath,dtype=np.uint8),-1),(100,100),interpolation=cv2.INTER_AREA)
    img=img.astype('float32')
    return np.array(img[:,:,:3]/255)

@app.route('/')
def index_page():
    return app.send_static_file('postPredict.html')
@app.route('/upload',methods=['POST'])
def anyname():
    model = load_model('static/resources/model_09-0.23-0.93.hdf5')
    file=request.files['file']
    imagefile=file.filename
    if file:
        dirfile='../static/resources/received_image'
        if not os.path.isdir(dirfile):
            os.makedirs(dirfile)
        imagefilepath=os.path.join(dirfile,imagefile)
        file.save(imagefilepath)
        inputs = getimage(imagefilepath)[np.newaxis, ...]
        predict_classname = classid(np.argmax(model.predict(inputs)[0]))
        print('对%s预测结果为:%s' % (imagefilepath, predict_classname))
        return jsonify(predict_classname=predict_classname)

if __name__=="__main__":
    app.run()