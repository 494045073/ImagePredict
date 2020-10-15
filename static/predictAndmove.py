import keras
import numpy as np
from keras.models import load_model
import cv2
import os
import shutil

def img_v(img_path):
    img = cv2.imread('JPEGImages/' + img_path)
    img = img.astype("float32")
    img /= 255.
    img = cv2.resize(img, dsize=(100, 100), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array(img[:, :, :3])

LABELS = ['apple', 'banana', 'orange', 'mixed']

if __name__ == "__main__":
    model = load_model("model_09-0.23-0.93.hdf5")
    print(model.summary())
    model.compile(loss=keras.losses.categorical_crossentropy,
                 optimizer=keras.optimizers.Adadelta(),
                 metrics=['accuracy'])
    path_file = 'flower/JPEGImages'
    copy_to = 'shuiguo'
    if not os.path.exists(copy_to):
        os.makedirs(copy_to)
    sur = os.listdir(path_file)
    for i in os.listdir(path_file):
        img = img_v(i)
        res = np.argmax(model.predict(np.array([img])))   #取得是标签值
        print(LABELS[res])
        if not os.path.exists(os.path.join(copy_to, LABELS[res])):
            os.makedirs(os.path.join(copy_to, LABELS[res]))
        for j in range(len(i)):
            shutil.copy('JPEGImages/' + i, os.path.join(copy_to, LABELS[res]))