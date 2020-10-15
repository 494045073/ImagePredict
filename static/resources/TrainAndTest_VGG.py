
# coding: utf-8

# In[1]:


from __future__ import print_function
import keras
import cv2
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense
from keras.models import Model
import glob
import numpy as np

import random
import os
import sys


# In[2]:


width=64
height=64
channel=3
train_ratio = 0.8
crop_fix_size = (60, 60)
crop_ratio = 0.5
lr=0.03
batch = 10
epoch=5
patienceEpoch=3

size  = width, height


# In[3]:


def CountFiles(path):
    files = []
    labels = []
    subdirs = os.listdir(path)
    print(subdirs)
    subdirs.sort()
    for index in range(len(subdirs)):
        subdir = os.path.join(path, subdirs[index])
        sys.stdout.flush()
        print("label --> dir : {} --> {}".format(index, subdir))
        for image_path in glob.glob("{}/*.jpg".format(subdir)):
            files.append(image_path)
            labels.append(index)

    hand_average = 82.1400742479
    return files, labels, len(subdirs), hand_average


# In[4]:


files, labels, clazz, average = CountFiles(r"flower")
c = list(zip(files, labels))
random.shuffle(c)
files, labels = zip(*c)

labels = np.array(labels)
labels = keras.utils.to_categorical(labels, clazz)


# In[5]:


def LoadImage(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_AREA)
    img = img.astype("float32")
    img -= average
    img /= 255.

    if random.random() < crop_ratio:
        im1 = img.copy()

        x = random.randint(0, img.shape[0] - crop_fix_size[0] - 1)
        y = random.randint(0, img.shape[1] - crop_fix_size[1] - 1)
        im1 = im1[x:x+crop_fix_size[0], y:y+crop_fix_size[1], :]
        im1 = cv2.resize(im1, dsize = size, interpolation = cv2.INTER_AREA)
        img = im1
    if random.random()<crop_ratio:
        im1 = img.copy()
        tmp=random.random()
        if tmp<0.3:
            im1 = cv2.flip(im1, 1, dst=None)  # 水平镜像
        elif tmp<0.6:
            im1 = cv2.flip(im1, 0, dst=None)  # 垂直镜像
        else:
            im1 = cv2.flip(im1, -1, dst=None)  # 对角镜像
        im1 = cv2.resize(im1, dsize=size, interpolation=cv2.INTER_AREA)
        img = im1
    return np.array(img)


# In[6]:


def LoadImageGen(files_r, labels_r, batch=32, label="label"):
    start = 0
    while start < len(files_r):
        stop = start + batch
        if stop > len(files_r):
            stop = len(files_r)
        imgs = []
        lbs = []
        import time
        mm = time.time()
        for i in range(start, stop):
            imgs.append(LoadImage(files_r[i]))
            lbs.append(labels_r[i])
        mm = time.time() - mm
        yield (np.array(imgs), np.array(lbs))
        if start + batch < len(files_r):
            start += batch
        else:
            c = list(zip(files_r, labels_r))
            random.shuffle(c)
            files_r, labels_r = zip(*c)
            start = 0
            # print("shuffled")


train_num = int(train_ratio * len(files))

train_x, train_y = files[:train_num], labels[:train_num]
test_x, test_y = files[train_num:], labels[train_num:]


# In[7]:


model_vgg16_conv = VGG16(weights=None,include_top=False, pooling='avg')


# In[8]:


input = Input(shape=(width, height, channel),name = 'image_input')


# In[9]:


output_vgg16_conv = model_vgg16_conv(input)

x = output_vgg16_conv

x = Dense(clazz, activation='softmax', name='predictions')(x)

#Create your own model
model = Model(inputs=input, outputs=x)


# In[10]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=lr, decay=0.),
              metrics=['accuracy'])


steps_per_epoch=int(len(train_x)/batch) - 1,
validation_steps = int(len(test_x) / batch) - 1,

tensorBoardCallBack = keras.callbacks.TensorBoard(log_dir="./tensorboard",
                                                  histogram_freq=0, write_graph=True,
                                                  write_grads=True, batch_size=batch,
                                                  write_images=True)

modelCheckpoint = ModelCheckpoint(filepath="./vggmodel_{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5", verbose = 0, save_best_only=False)
print("\nclass num : {}, pic average: {}, train num: {}, test num: {}, batch : {}, train steps: {}, validation steps : {}".format(clazz, average, len(train_x), len(test_x), batch, steps_per_epoch, validation_steps))
model.fit_generator(
    LoadImageGen(train_x, train_y, batch=batch, label = "train"),
    steps_per_epoch=int(len(train_x)/batch),
    epochs = epoch,
    verbose = 1,
    validation_data=LoadImageGen(test_x, test_y, batch=batch, label="test"),
    validation_steps = int(len(test_x) / batch),
    # callbacks=[(EarlyStopping(monitor='val_acc', patience=patienceEpoch)), tensorBoardCallBack, modelCheckpoint],
)


# In[11]:
model_savePaths = 'f_l.h5'
model.save(model_savePaths)

score = model.evaluate_generator(
    LoadImageGen(test_x, test_y, batch=batch),
    steps = int(len(test_x) / batch)
)
print("Test loss: {}, Test accuracy: {}".format(score[0], score[1]))

