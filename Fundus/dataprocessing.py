import os
import argparse
import sys
import time
import random
import keras
import cv2
import numpy as np

def crop(im, ratio):
    new_width = int(im.shape[1] * ratio)
    new_height = int(im.shape[0] * ratio)

    left_margin = (im.shape[1] - new_width) // 2
    top_margin = (im.shape[0] - new_height) // 2
    return im[top_margin:-top_margin,
            left_margin:-left_margin]

def rgb2gray(im):
    return np.dot(im[...,:3], [0.2989, 0.5870, 0.1140])

def clahe_image(im, clip_limit=2.0, title_grid_size=(8,8)):
  if len(im.shape) == 2:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=title_grid_size)
    return clahe.apply(im)
  else:
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def image_preprocessing(im, rescale, resize_factor):
    res = crop(im, 0.7)
    
    ## 이미지 크기 조정 및 픽셀 범위 재설정
    h, w, c = 3072, 3900, 3
    nh, nw = int(h//resize_factor), int(w//resize_factor)
    # print(im.shape)

    res = cv2.resize(res, (nw, nh), interpolation=cv2.INTER_AREA)

    gray = rgb2gray(res)
    gray = gray.astype(np.uint8)
    gray = clahe_image(gray)

    gray = np.expand_dims(gray, axis=2)
    res = np.concatenate((res,gray), axis=2)

    if rescale == True:
        res = res / 255.

    return res


def Label2Class(label):     # one hot encoding (0-3 --> [., ., ., .])

    resvec = [0, 0, 0, 0]
    if label == 'AMD':		cls = 1;    resvec[cls] = 1
    elif label == 'RVO':	cls = 2;    resvec[cls] = 1
    elif label == 'DMR':	cls = 3;    resvec[cls] = 1
    else:					cls = 0;    resvec[cls] = 1		# Normal

    return resvec


def dataset_loader(img_path, rescale, resize_factor):

    t1 = time.time()
    print('Loading training data...\n')
    if not ((resize_factor == 1.) and (rescale == False)):
        print('Image preprocessing...')
    if not resize_factor == 1.:
        print('Image size is 3072*3900*3')
        print('Resizing the image into {}*{}*{}...'.format(int(3072//resize_factor), int(3900//resize_factor), 3))
    if not rescale == False:
        print('Rescaling range of 0-255 to 0-1...\n')

    ## 이미지 읽기
    p_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(img_path) for f in files if all(s in f for s in ['.jpg'])]
    p_list.sort()
    num_data = len(p_list)

    labels = []
    h, w, c = 3072, 3900, 3
    nh, nw = int(h//resize_factor), int(w//resize_factor)
    images = np.empty((len(p_list), nh, nw, 4), dtype=np.float32)
    for i, p in enumerate(p_list):
        im = cv2.imread(p, 3)
        im = image_preprocessing(im, rescale=rescale, resize_factor=resize_factor)
        images[i, ...,] = im

        # label 데이터 생성
        l = Label2Class(p.split('/')[-2])
        labels.append(l)
        
        if (i + 1) % 100 == 0:
            print(i + 1, '/', num_data, ' image(s)')

    labels = np.array(labels)

    t2 = time.time()
    print('Dataset prepared for' ,t2 -t1 ,'sec')
    print('Images:' ,images.shape ,'np.array.shape(files, views, width, height)')
    print('Labels:', labels.shape, ' among 0-3 classes')

    return images, labels


