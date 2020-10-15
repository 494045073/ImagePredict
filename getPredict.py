import os
import cv2
import json
import random
import numpy as np
from keras.models import load_model
from flask import Flask

app=Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('getPredict.html')

def classname(classid):
    return ['apple','banana','orange','mixed'][classid]
def getImage(imagepath):
    img=cv2.resize(cv2.imdecode(np.fromfile(imagepath,dtype=np.uint8),-1),(100,100),interpolation=cv2.INTER_AREA)
    img = img.astype('float32')
    return np.array(img[:,:,:3]/255.)
@app.route('/download/',methods=['GET'])
def downImage():
    model=load_model('static/resources/model_09-0.23-0.93.hdf5',compile=False)
    path='static/predict/'
    folders=os.listdir(path)
    filelist={}
    # cq_num=0
    # ch_num=0
    for folder in folders:
        filelist[folder]=[]
        folder_file=os.path.join(path,folder)
        if os.listdir(folder_file):
            fileall=os.listdir(folder_file)
            files=random.sample(fileall,5)
            for file in files:
                # cq_num+=1
                imagefile=os.path.join(folder_file,file)
                inputs=getImage(imagefile)[np.newaxis,...]
                predict_classname=classname(np.argmax(model.predict(inputs)[0]))
                # if predict_classname==folder:
                #     ch_num+=1
                print('对%s预测结果:%s'%(imagefile,predict_classname))
                filelist[folder].append({'filename':file,'path':imagefile,'type':predict_classname,'archetype':folder})
    # zql=ch_num/cq_num
    # print('预测数:%s'%(cq_num))
    # print('正确数:%s'%(ch_num))
    # print('准确率:%s'%(zql))
    return json.dumps(filelist)

if __name__=='__main__':
    app.run()