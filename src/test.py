# -*- coding: utf-8 -*-
from __future__ import print_function

from evaluate import infer
from DB import Database
from color import Color
from gabor import Gabor
from HOG import HOG
import shutil
import cv2
from skimage import io
from matplotlib import pyplot as plt
# import argparse module for argument parsing
import argparse
import math

depth = 10
d_type = 'd1'
# query_idx = 53
# initialize the argument parser
base_dir = 'D:/Learn school/XuLyAnh/CBIR/'
ap = argparse.ArgumentParser()

# add argument for query image path
ap.add_argument("-q", "--query_idx", required=True, help="index query")
# add argument for feature class
ap.add_argument("-c", "--class", required=True, help="Feature class")
# parse the arguments
args = vars(ap.parse_args())

if __name__ == '__main__':
    db = Database()

    # retrieve by color
    method = Color()
    samples = method.make_samples(db)
    query = samples[int(args["query_idx"])]
    _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    print(result)
    # lst = []
    # img_class = []
    # for i in range(0, depth):
    #     idx = result[i]
    #     name_img = idx['img']
    #     class_name = idx['cls']
    #     lst.append(name_img)
    #     img_class.append(class_name)
    # print(lst)
    # print(img_class)
    #
    # for i in range(len(lst)):
    #     shutil.copy(lst[i], "result")
    # # create figure
    # fig = plt.figure(figsize=(10, 7))
    #
    # # setting values to rows and column variables
    # rows = 1
    # columns = depth + 2
    #
    # Image_query = io.imread(query['img'])
    # fig.add_subplot(rows, columns, 1)
    #
    # # showing image
    # plt.imshow(Image_query)
    # plt.axis('off')
    # plt.title("Query")
    #
    # for i in range(1, depth + 1):
    #     # Adds a subplot at the 1st position
    #     fig.add_subplot(rows, columns, i + 1)
    #     Image = io.imread(lst[i - 1])
    #     plt.imshow(Image)
    #
    #     plt.axis('off')
    #     plt.title("a")
    #     if img_class[i - 1] == query['cls']:
    #         plt.title('ID:' + str(img_class[i - 1]), color='green')
    #     else:
    #         plt.title('ID:' + str(img_class[i - 1]), color='red')
    #
    # fig.savefig("show.png")
    lst = []
    img_class = []
    for i in range(0, depth):
        idx = result[i]
        name_img = idx['img']
        class_name = idx['cls']
        lst.append(name_img)
        img_class.append(class_name)
    print(lst)
    print(img_class)
    # setting values to rows and column variables
    rows = int(math.modf(math.sqrt(depth))[1] + 1)
    columns = int(math.modf(math.sqrt(depth))[1] + 1)
    query_image = cv2.imread(base_dir + query['img'])
    fig1 = plt.figure(figsize=(20, 15))
    fig1.add_subplot(rows+1, columns, 1)
    # showing image
    plt.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    title1 = "ẢNh truy vấn - class {} ".format(query['cls'])
    plt.title(title1)

    # create figure
    # fig = plt.figure(figsize=(20, 15))
    title_s = "Kết quả truy vấn theo đặc trưng {}".format(args["class"])
    fig1.suptitle(title_s, fontsize=14, fontweight='bold')

    for i in range(len(result)):
        img = cv2.imread(base_dir + result[i]['img'])
        fig1.add_subplot(rows, columns, i + 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        # plt.title(result[i]['cls'])
        if img_class[i - 1] == query['cls']:
            title2 = "ID: {}, dis: {}".format(str(img_class[i - 1]),result[i]['dis'])
            # plt.title('ID:' + str(img_class[i - 1]), color='green')
            plt.title(title2, color = 'green')
        else:
            #plt.title('ID:' + str(img_class[i - 1]), color='red')
            title3 = "ID: {}, dis: {}".format(str(img_class[i - 1]), result[i]['dis'])
            plt.title(title3, color='red')

    fig1.savefig("show.png")

img = cv2.imread("show.png")

cv2.imshow('Display Image Result', img)
cv2.waitKey(0)
# python src/test.py --query_idx 53
# python src/test.py --query_idx 10 --c color