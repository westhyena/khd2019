import os
import argparse
import sys
import time
import keras
import cv2
import numpy as np

from sklearn.metrics import accuracy_score


from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import BatchNormalization, ReLU
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils.training_utils import multi_gpu_model
import keras.backend.tensorflow_backend as K
import nsml
from nsml.constants import DATASET_PATH, GPU_NUM

from model import cnn_sample, inception_v3, efficientnet, resnext_50, densenet_121
from dataprocessing import image_preprocessing, dataset_loader


## setting values of preprocessing parameters
RESIZE = 10.
RESCALE = True


def bind_model(inception_model, inception_ratio,
               efficient_model, efficient_ratio,
               resnet_model, resnet_ratio,
               densenet_model, densenet_ratio
               ):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        inception_model.save_weights(os.path.join(dir_name, 'inception_model'))
        efficient_model.save_weights(os.path.join(dir_name, 'efficient_model'))
        resnet_model.save_weights(os.path.join(dir_name, 'resnet_model'))
        densenet_model.save_weights(os.path.join(dir_name, 'densenet_model'))
        # model.save_weights(file_path,'model')
        print('model saved!')

    def load(dir_name):
        inception_model.load_weights(os.path.join(dir_name, 'inception_model'))
        efficient_model.load_weights(os.path.join(dir_name, 'efficient_model'))
        resnet_model.load_weights(os.path.join(dir_name, 'resnet_model'))
        densenet_model.load_weights(os.path.join(dir_name, 'densenet_model'))
        print('model loaded!')

    def infer(data, rescale=RESCALE, resize_factor=RESIZE):  ## test mode
        ##### DO NOT CHANGE ORDER OF TEST DATA #####
        X = []
        for i, d in enumerate(data):
            # test 데이터를 training 데이터와 같이 전처리 하기
            X.append(image_preprocessing(d, rescale, resize_factor))
        X = np.array(X)

        inception_pred = inception_model.predict(X)
        efficient_pred = efficient_model.predict(X)
        resnet_pred = resnet_model.predict(X)
        densenet_pred = densenet_model.predict(X)

        pred = (inception_pred * inception_ratio + efficient_pred * efficient_ratio)
        pred += (resnet_pred * resnet_ratio + densenet_pred * densenet_ratio)
        pred = np.argmax(pred, axis=1)
        print('Prediction done!\n Saving the result...')
        return pred

    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--num_classes', type=int, default=4)                     # DO NOT CHANGE num_classes, class 수는 항상 4
    args.add_argument('--load_from', type=str, default=None)
    
    args.add_argument('--inception', type=str, default='')
    args.add_argument('--inception_ckpt', type=str, default='')
    args.add_argument('--inception_ratio', type=float, default=1.)

    args.add_argument('--efficient', type=str, default='')
    args.add_argument('--efficient_ckpt', type=str, default='')
    args.add_argument('--efficient_ratio', type=float, default=1.)

    args.add_argument('--resnet', type=str, default='')
    args.add_argument('--resnet_ckpt', type=str, default='')
    args.add_argument('--resnet_ratio', type=float, default=1.)

    args.add_argument('--densenet', type=str, default='')
    args.add_argument('--densenet_ckpt', type=str, default='')
    args.add_argument('--densenet_ratio', type=float, default=1.)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    config = args.parse_args()

    seed = 1234
    np.random.seed(seed)

    def nsml_load(dir_name):
        model.load_weights(os.path.join(dir_name, 'model'))
        print('model loaded!')

    """ Model """
    h, w = int(3072//RESIZE), int(3900//RESIZE)
    input_shape = (h, w, 4)
    num_classes = config.num_classes

    inception_model = inception_v3(input_shape, num_classes)
    efficient_model = efficientnet(input_shape, num_classes)
    resnet_model = resnext_50(input_shape, num_classes)
    densenet_model = densenet_121(input_shape, num_classes)
    
    learning_rate = 1e-4
    sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    inception_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])
    efficient_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])
    resnet_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])
    densenet_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])

    def inception_load(dir_name):
        inception_model.load_weights(os.path.join(dir_name, 'model'))
        print('inception loaded')
    
    def efficient_load(dir_name):
        efficient_model.load_weights(os.path.join(dir_name, 'model'))
        print('efficient loaded')

    def resnet_load(dir_name):
        resnet_model.load_weights(os.path.join(dir_name, 'model'))
        print('resnet loaded')

    def densenet_load(dir_name):
        densenet_model.load_weights(os.path.join(dir_name, 'model'))
        print('densenet loaded')
    
    nsml.load(checkpoint=config.inception_ckpt, load_fn=inception_load, session=config.inception)
    nsml.load(checkpoint=config.efficient_ckpt, load_fn=efficient_load, session=config.efficient)
    nsml.load(checkpoint=config.resnet_ckpt, load_fn=resnet_load, session=config.resnet)
    nsml.load(checkpoint=config.densenet_ckpt, load_fn=densenet_load, session=config.densenet)
    
    bind_model(inception_model, config.inception_ratio,
               efficient_model, config.efficient_ratio,
               resnet_model, config.resnet_ratio,
               densenet_model, config.densenet_ratio)
    if config.pause:  ## test mode일 때
        print('Inferring Start...')
        nsml.paused(scope=locals())

    if config.mode == 'train':  ### training mode일 때
        print('Training Start...')

        img_path = DATASET_PATH + '/train/'

        if config.load_from:
            # Load From Saved Session
            data = {}
            def nsml_load(dir_path, **kwargs):
                images = np.load(os.path.join(dir_path, 'data_x.npy'))
                labels = np.load(os.path.join(dir_path, 'data_y.npy'))
                data['x'] = images
                data['y'] = labels
                print("Data Loaded!!!")
            nsml.load(checkpoint='data', load_fn=nsml_load, session=config.load_from)

            print("Data Loaded???")
            print(type(data['x']))
            print(type(data['y']))
            images = data['x']
            labels = data['y']
        else:
            images, labels = dataset_loader(img_path, resize_factor=RESIZE, rescale=RESCALE)
            # containing optimal parameters

            def nsml_save(dir_path, **kwargs):
                np.save(os.path.join(dir_path, 'data_x.npy'), images)
                np.save(os.path.join(dir_path, 'data_y.npy'), labels)
                print("Data saved!!!")
            nsml.save(checkpoint='data', save_fn=nsml_save)

        ## data 섞기
        dataset = [[X, Y] for X, Y in zip(images, labels)]
        seed = 1234
        np.random.shuffle(dataset)
        X = np.array([n[0] for n in dataset])
        Y = np.array([n[1] for n in dataset])

        """ Training loop """
        t0 = time.time()

        ## data를 trainin과 validation dataset으로 나누기
        train_val_ratio = 0.8
        tmp = int(len(Y)*train_val_ratio)
        X_train = X[:tmp]
        Y_train = Y[:tmp]
        X_val = X[tmp:]
        Y_val = Y[tmp:]

        inception_pred = inception_model.predict(X_val)
        efficient_pred = efficient_model.predict(X_val)
        resnet_pred = resnet_model.predict(X_val)
        densenet_pred = densenet_model.predict(X_val)

        pred = (inception_pred * config.inception_ratio + efficient_pred * config.efficient_ratio)
        pred += (resnet_pred * config.resnet_ratio + densenet_pred * config.densenet_ratio)
        pred = np.argmax(pred, axis=1)
        y_true = np.argmax(Y_val, axis=1)
        
        acc = accuracy_score(y_true, pred)
        nsml.report(summary=True, step=0, epoch_total=0, val_acc=acc)
        nsml.save('zero')
        print(acc)
        # print(model.predict_classes(X))



