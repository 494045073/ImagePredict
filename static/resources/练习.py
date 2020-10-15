import cv2
import os
import glob
import numpy as np
import keras
import random
import tensorflow as tf
import pickle
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense
from keras.models import Model

# a = cv2.imread('Black_Footed_Albatross_0001_796111.jpg')
# # cv2.imshow("a",a)
# # cv2.waitKey(0)
# b = cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
# cv2.imshow("b",b)
# cv2.waitKey(0)
# c = cv2.cvtColor(a,cv2.COLOR_BGR2HSV)
# cv2.imshow("c",c)
# cv2.waitKey(0)

# d = os.listdir('images')
# print(d)
# for i in range(len(d)):
#     files = []
#     label = []
#     f = os.path.join('images',d[i])
#     print(f)
#     for i_images in glob.glob("{}/*.jpg".format(d)):
#         files.append(i_images)
#         label.append(i)

# Width = 64
# Height = 64
# img_a = 0.5
# clazz = 3
# img_ratio = 0.8
# lr = 0.03
# bacth = 15
# epoch = 10
# size = Width,Height
# pacg = 'images'
#
# def data1(images):
#     sudir = os.listdir(images)
#     file = []
#     print(sudir)
#     sudir.sort()
#     for index in sudir:
#         sudir = os.path.join(images, index)
#         print("wenjian{}".format(sudir))
#         file.append(sudir)
#     return file,len(sudir)
# file,clazz = data1(r"abc")
#
# def lab(image):
#     sudir1 = os.listdir(image)
#     sudir1.sort()
#     lable = []
#     for i in range(len(sudir1)):
#         lable.append(i)
#     return lable
# lable = lab(r"abc")
# lable = np.array(lable)

# def image1(image):
#     img = cv2.imread(image)
#     img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_AREA)
#     img = img.astepe("float 32")
#     img /= 255.
#
#     if img is not None:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img,dsize=size,interpolation=cv2.INTER_NEAREST)
#     if random.random() < img_a:
#         tp = random.random()
#         if tp < 0.3 :
#             img = cv2.flip(img, 1, dst=None)
#         elif tp < 0.6:
#             img = cv2.flip(img, 0, dst=None)
#         else:
#             img = cv2.flip(img, -1, dst=None)
#         img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_AREA)
#     return np.array(img)



# train_num = int(len(file) * img_ratio)


# train_x, train_y = file[:train_num], lable[:train_num]
# test_x, test_y = file[train_num:],lable[train_num:]
# # train_x /= 255.
# # test_x /= 255.
#
# x_inpit = tf.placeholder(tf.float32,shape=[None,])
# y_input = tf.placeholder(tf.float32,shape=[None,])
# print(x_inpit)
# print(y_input)
# loss = tf.losses.softmax_cross_entropy(train_y,)


def image(images):
    file = os.listdir(images)
    files = []
    file.sort()
    for i in file:
        filer = i
        files.append(filer)
    return files
files = image("flower")
print(files)
data_a = 'flower'

def imgst(path_data, data_i = 0.8, resize = True, data_f = None):
    file_name = os.path.join(data_a, path_data + str(width) + "X" + str(height) + ".pkl")
    train_x = []
    train_y = []
    text_x = []
    text_y = []
    lable = 0
    pic_dir  = image(r"flower")
    for i in pic_dir:
        if not os.path.isdir(os.path.join(data_a, i)):
            continue
        pic_set = image(os.path.join(data_a,i))
        train_num = int(len(pic_set)*data_i)
        train_index = 0
        for pic_index in pic_set:
            if not os.path.isfile(os.path.join(data_a, i, pic_index)):
                continue
            img = cv2.imread(os.path.join(data_a, i, pic_index))
            if img is None:
                continue
            img = img.astype("float32")
            img /= 255.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if (resize):
                img = cv2.resize(img, (64,64))
            if (data_f == "channels_last"):
                img = img.reshape(-1,64,64,3)
            elif (data_f == "channels_first"):
                img = img.reshape(-1,3,64,64)
            if (train_index < train_num):
                train_x.append(img)
                train_y.append(lable)
            else:
                text_x.append(img)
                text_y.append(lable)
            train_index += 1
        if len(pic_set) != 0:
            lable += 1
    train_x = np.concatenate(train_x, axis=0)
    text_x = np.concatenate(text_x, axis=0)
    train_y = np.array(train_y)
    text_y = np.array(text_y)
    pickle.dump([(train_x, train_y), (text_x, text_y)], open(file_name, "wb"))
    return (train_x, train_y), (text_x, text_y)

def main():
    global width, height
    width = 64
    height = 64
    lr = 0.03
    num_clazz = 2
    channel = 3
    (train_x, train_y), (text_x, text_y) = imgst("flower", 0.8, data_f="channels_last")
    # # train_x /= 255.
    # text_x /= 255.
    print(train_x.shape)
    print(text_x.shape)
    train_y = keras.utils.to_categorical(train_y, num_clazz)
    text_y = keras.utils.to_categorical(text_y, num_clazz)
    model_vgg16_conv = VGG16(weights=None, include_top=False, pooling='avg')
    # %%

    input = Input(shape=(width, height, channel), name='image_input')

    # %%

    output_vgg16_conv = model_vgg16_conv(input)

    x = output_vgg16_conv

    x = Dense(num_clazz, activation='softmax', name='predictions')(x)

    # Create your own model
    model = Model(inputs=input, outputs=x)

    # %%
    modelcheck = ModelCheckpoint(filepath="./model_{epoch:02d}-{loss:.2f}-{accuracy:.2f}.hdf5", verbose=0,
                                 save_best_only=False)
    model_list = [modelcheck]

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(lr=lr, decay=0.),
                  metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=7, batch_size=1,callbacks=model_list )
    print('\nTesting ------------')  # 对测试集进行评估，额外获得metrics中的信息
    loss, accuracy = model.evaluate(text_x, text_y)
    print('\n')
    print('test loss: ', loss)
    print('test accuracy: ', accuracy)

    model_savePaths = 'flow_los.h5'
    model.save(model_savePaths)


if __name__ == '__main__':
    main()

