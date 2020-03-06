import numpy as np
import argparse
import cv2
import math
import os
import time
from threading import Thread
from os import listdir, makedirs
from os.path import isfile, join, basename, splitext, exists

def vid2img(pathTofolder, outputPath, fn):
        #this is the folder for letter a # needs to be changed for google drive
        video = cv2.VideoCapture(join(pathTofolder,fn))
        count = 0
        success,image = video.read()
        success = True
        new_img_folder = splitext(fn)[0]
        if new_img_folder == 'aa' or 'ah':
                new_img_folder == 'a'
        elif new_img_folder == 'eh':
                new_img_folder = 'e'
        elif new_img_folder == 'ii':
                new_img_folder = 'i'
        elif new_img_folder == 'oh':
                new_img_folder == 'o'
        elif new_img_folder == 'uh':
                new_img_folder = 'u'


        new_path = outputPath + new_img_folder
        if not exists(new_path):
                os.makedirs(new_path)
        while success:
                success,image = video.read()
                if success == False:
                        break
                print ('Read a new frame: ', success)
                cv2.imwrite(new_path + '\\image%d.jpg' % count, image) #are we gonna go with jpg or png, I think both work
                count += 1


def executeFrameConversion(pathTofolder, outputPath, fn):
        thread = Thread(target = vid2img, args = (pathTofolder, outputPath, fn))
        # t1 = Thread(target = vid2img, args = ('C:\\Users\\anish\\Desktop\\SiemensStuff\\w & space', 'C:\\Users\\anish\\Desktop\\SiemensStuff\\train_data\\raw_imgs_1440x1080\\train_data_raw_4\\', 1))
        # t2 = Thread(target = vid2img, args = ('C:\\Users\\anish\\Desktop\\SiemensStuff\\w & space', 'C:\\Users\\anish\\Desktop\\SiemensStuff\\train_data\\raw_imgs_1440x1080\\train_data_raw_4\\', 2))
        # t3 = Thread(target = vid2img, args = ('C:\\Users\\anish\\Desktop\\SiemensStuff\\w & space', 'C:\\Users\\anish\\Desktop\\SiemensStuff\\train_data\\raw_imgs_1440x1080\\train_data_raw_4\\', 2))
        # t4 = Thread(target = vid2img, args = ('C:\\Users\\anish\\Desktop\\SiemensStuff\\w & space', 'C:\\Users\\anish\\Desktop\\SiemensStuff\\train_data\\raw_imgs_1440x1080\\train_data_raw_4\\', 2))
        # t5 = Thread(target = vid2img, args = ('C:\\Users\\anish\\Desktop\\SiemensStuff\\w & space', 'C:\\Users\\anish\\Desktop\\SiemensStuff\\train_data\\raw_imgs_1440x1080\\train_data_raw_4\\', 2))
        # t1.start()
        # t2.start()
        return thread

def main(pathTofolder, outputPath):
        threads = [executeFrameConversion(pathTofolder, outputPath, fn) for fn in listdir(pathTofolder)]
        for t in threads:
                t.start()
        for t in threads:
                t.join()

if __name__ == '__main__':
        main(pathTofolder = 'C:\\Users\\anish\\Desktop\\SiemensStuff\\a_n_d\\a_n_d', outputPath = 'C:\\Users\\anish\\Desktop\\SiemensStuff\\a_n_d\\a_n_d_\\')