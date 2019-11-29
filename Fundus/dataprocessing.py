import os
import argparse
import sys
import time
import random
import keras
import cv2
import numpy as np

def crop(im, ratio, center_pos):
    return im
    

def resize(im, new_width, new_height):
    return cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_AREA)


def clahe(im, clip_limit=2.0, title_grid_size=(8,8)):
    if len(im.shape) == 2:          # is gray scale
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=title_grid_size)
        return clahe.apply(im)
    else:
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def image_preprocessing(im, resize_width, resize_height,
                        crop_ratio=1., crop_center_pos=(0.5, 0.5),
                        apply_clahe=False, rescale=True):
    # crop
    res = crop(im, crop_ratio, crop_center_pos)

    # resize
    res = resize(res, resize_width, resize_height)

    # clahe
    res = clahe(res)

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


def dataset_loader(img_path, **kwargs):

    t1 = time.time()
    print('Loading training data...\n')
    # if not ((resize_factor == 1.) and (rescale == False)):
    #     print('Image preprocessing...')
    # if not resize_factor == 1.:
    #     print('Image size is 3072*3900*3')
    #     print('Resizing the image into {}*{}*{}...'.format(int(3072//resize_factor), int(3900//resize_factor), 3))
    # if not rescale == False:
    #     print('Rescaling range of 0-255 to 0-1...\n')

    ## 이미지 읽기
    p_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(img_path) for f in files if all(s in f for s in ['.jpg'])]
    p_list.sort()
    num_data = len(p_list)

    images = []
    labels = []

    image_height = kwargs["resize_height"]
    image_width = kwargs["resize_width"]
    image_channel = 3
    # if kwargs["apply_clahe"]:
    #     image_channel += 1

    images = np.empty((len(p_list), image_height, image_width, image_channel), dtype=np.float32)
    for i, p in enumerate(p_list):
        im = cv2.imread(p, 3)
        im = image_preprocessing(im, **kwargs)
        images[i, ...,] = im

        # label 데이터 생성
        l = Label2Class(p.split('/')[-2])
        labels.append(l)

        # print(i + 1, '/', num_data, ' image(s)')

    labels = np.array(labels)

    t2 = time.time()
    print('Dataset prepared for' ,t2 -t1 ,'sec')
    print('Images:' ,images.shape ,'np.array.shape(files, views, width, height)')
    print('Labels:', labels.shape, ' among 0-3 classes')

    return images, labels


