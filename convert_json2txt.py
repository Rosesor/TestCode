# -*-coding:utf-8-*-
#
#  test.py
#  DataTest
#
#  Created by zhoujianwen on 2019/5/1.
#  Copyright © 2019年 Clement. All rights reserved.
#

from PIL import Image
import os
import cPickle
import json
import glob
import io
import shutil
import sys
from scipy.spatial import distance as dist
import numpy as np
import math

reload(sys)
sys.setdefaultencoding('utf-8')
from matplotlib.path import Path


def format_output():
    res = open(os.path.join('./', 'test.txt'), 'w')
    for i in range(len(lines) - 2):
        res.write(",".join(str(j) for j in lines[i]))
    res.write(",".join(str(i) for i in lines[len(lines) - 1]))
    res.write('\n')
    res.close()


def check_points():
    lines = []
    index = 0
    labelme_json = glob.glob('/media/wyu/software/new_image/300_json/*.json')
    for num, json_file in enumerate(labelme_json):
        with open(json_file, 'r') as fp:
            data = json.load(fp)
            for shapes in data['shapes']:
                if len(shapes['points']) % 2 or not len(shapes['points']) % 6 or not len(shapes['points']) % 8:
                    index = index + 1
                    lines.append(str.format('id:{0},{1}.jpg,坐标数量：{2}\n', index, json_file.split('/')[-1].split('.')[0],
                                            len(shapes['points'])))
    f = open(os.path.join('./', 'error_log.txt'), 'w')
    for i in lines:
        f.write(str(i))
    f.close()


def check_label_len(num):
    lines = []
    index = 0
    path = '/media/wyu/software/new_image/300_json/*.json'  # './data/train/*.json'
    labelme_json = glob.glob(path)
    for num, json_file in enumerate(labelme_json):
        with open(json_file, 'r') as fp:
            data = json.load(fp)
            for shapes in data['shapes']:
                if len(shapes['label']) > 1 and len(shapes['label']) <= 4 or len(
                        shapes['label'].strip()) == 0:  # =1 is char,=0 or =2 is error,>2 is datatime
                    print (data['imagePath'], shapes['label'])
                    index = index + 1
                    lines.append(str.format('id:{0},{1},{2},标签长度：{3}\n', index, data['imagePath'], shapes['label'],
                                            len(shapes['label'])))

    outpath = './'
    f = open(os.path.join(outpath, 'error_log2.txt'), 'w')
    for i in lines:
        f.write(str(i))
    f.close()


class biaozhu(object):
    def __init__(self, label, points, ds):
        self.label = label
        self.points = points
        self.ds = ds
        self.charbox = []

    def __repr__(self):
        return repr((self.label, self.points, self.ds, self.charbox))


def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    print (a, b)
    print (zip(a, b))
    for a1, b1 in zip(a, b):
        part_up += a1 * b1
        a_sq += a1 ** 2
        b_sq += b1 ** 2
    part_down = math.sqrt(a_sq * b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down


# this function is confined to rectangle
def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


class biaozhubox(object):
    def __init__(self, label, points, ds, ce):
        self.label = label
        self.points = points
        self.ds = ds
        self.charbox = []
        self.ce = ce

    def __repr__(self):
        return repr((self.label, self.points, self.ds, self.charbox, self.ce))


def json2txt():
    file_path = 'F:/cocoDataAugment/data/rerotate_resize/*.json'  # 存储.txt的地方
    # file_path = './100/*.json'
    labelme_json = glob.glob(file_path)
    print(labelme_json)
    for num, json_file in enumerate(labelme_json):
        print(json_file)
        with open(json_file, 'r') as fr:
            data = json.load(fr)
            boxs = []
            index = 0
            for shapes in data['shapes']:  # 统计检测框数量，父级检测框boxs
                if len(shapes['label'].strip()) > 1:
                    if len(shapes['points']) == 2:
                        rect = shapes['points']
                        x1_box = rect[1][0] if rect[0][0] > rect[1][0] else rect[0][0]  # min_x
                        y1_box = rect[1][1] if rect[0][1] > rect[1][1] else rect[0][1]  # min_y
                        x3_box = rect[0][0] if rect[0][0] > rect[1][0] else rect[1][0]  # max_x
                        y3_box = rect[0][1] if rect[0][1] > rect[1][1] else rect[1][1]  # max_y
                        x2_box, y2_box = [x3_box, y1_box]
                        x4_box, y4_box = [x1_box, y3_box]
                    else:
                        print(data['imagePath'])
                        polygons = order_points(np.array(shapes['points']))
                        point = polygons
                        x1_box, y1_box = point[0]
                        x2_box, y2_box = point[1]
                        x3_box, y3_box = point[2]
                        x4_box, y4_box = point[3]

                    ds = np.sqrt(np.square(0 - x1_box) + np.square(0 - y1_box))
                    ce_x = (x1_box + x3_box) / 2
                    ce_y = (y1_box + y3_box) / 2
                    label = shapes['label'].replace(' ', '')
                    boxs.append(
                        biaozhubox(label, [[x1_box, y1_box], [x2_box, y2_box], [x3_box, y3_box], [x4_box, y4_box]], ds,
                                   [ce_x, ce_y]))

            for box in boxs:  # 父级检测框关联字符框boxs
                # 获取父级检测框的坐标
                x1_box, y1_box = [float(i) for i in box.points[0]]
                x2_box, y2_box = [float(i) for i in box.points[1]]
                x3_box, y3_box = [float(i) for i in box.points[2]]
                x4_box, y4_box = [float(i) for i in box.points[3]]
                for shapes in data['shapes']:  # 添加字符框char
                    label = shapes['label'].replace(' ', '')
                    if len(label) > 1:
                        continue
                    if len(shapes['points']) == 2:
                        rect = shapes['points']
                        x1 = rect[1][0] if rect[0][0] > rect[1][0] else rect[0][0]  # min_x
                        y1 = rect[1][1] if rect[0][1] > rect[1][1] else rect[0][1]  # min_y
                        x3 = rect[0][0] if rect[0][0] > rect[1][0] else rect[1][0]  # max_x
                        y3 = rect[0][1] if rect[0][1] > rect[1][1] else rect[1][1]  # max_y
                        x2, y2 = [x3, y1]
                        x4, y4 = [x1, y3]
                    else:
                        print (data['imagePath'])
                        polygons = order_points(np.array(shapes['points']))
                        point = polygons.astype(np.float)
                        x1, y1 = point[0]
                        x2, y2 = point[1]
                        x3, y3 = point[2]
                        x4, y4 = point[3]
                    ds = np.sqrt(np.square(x1_box - x1) + np.square(y1_box - y1))
                    ce_x = (x1 + x3) / 2
                    ce_y = (y1 + y3) / 2
                    # ds = format((x1+y1+x2+y2+x3+y3+x4+y4)/8,'.2f')
                    points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    # min_X = np.amin(point,axis=0)[0]
                    # min_Y = np.amin(point,axis=1)[1]
                    # max_X = np.amax(point,axis=0)[0]
                    # max_Y = np.amax(point,axis=1)[1]
                    if len(boxs) == 1:
                        box.charbox.append(biaozhubox(label, points, ds, [ce_x, ce_y]))
                    else:  # 图片超过1个父级标签就执行下面语句
                        if (
                        Path([(x1_box, y1_box), (x2_box, y2_box), (x3_box, y3_box), (x4_box, y4_box)]).contains_points(
                                [(ce_x, ce_y)])):
                            box.charbox.append(biaozhubox(label, points, ds, [ce_x, ce_y]))
                        # else:
                        #     print data['imagePath']  #输出漏检的字符框

                # for i in range(len(box.charbox)):
                #     for j in range(len(box.charbox) - 1 - i):
                #         if box.charbox[j].ds > box.charbox[j + 1].ds:
                #             box.charbox[j], box.charbox[j + 1] = box.charbox[j + 1], box.charbox[j]
                box.charbox = sorted(box.charbox, key=lambda biaozhubox: biaozhubox.ds)
                # if box.charbox[0].label not in box.label[0] and box.charbox[-1].label not in box.label[-1]:
                #     box.charbox = sorted(box.charbox, key=lambda biaozhubox: biaozhubox.ds,reverse=True)
                # for i in box.charbox:
                #     if i.label.find('合') >=0 or i.label.find('格')>=0:
                #         box.charbox.remove(i)
                # index = index + 1
                # print index
            # 父级标签和子级标签关联完就按照指定格式输出到txt文件
            (filepath, tempfilename) = os.path.split(file_path)
            # print(filepath)
            # (filename, extension) = os.path.splitext(tempfilename)
            with open(os.path.join(filepath + '/', str.format('{0}.txt', data['imagePath'])),
                      'w') as fw:  # 把相关数据写入.txt文件
                for box in boxs:
                    x1_box, y1_box = box.points[0]
                    x2_box, y2_box = box.points[1]
                    x3_box, y3_box = box.points[2]
                    x4_box, y4_box = box.points[3]
                    if box.label.find('合') >= 0 or box.label.find('格') >= 0:
                        box.label = box.label.replace('合', '').replace('格', '')
                    fw.write(
                        str.format('{0},{1},{2},{3},{4},{5},{6},{7},{8},', x1_box, y1_box, x2_box, y2_box, x3_box,
                                   y3_box, x4_box, y4_box, box.label))
                    for j in range(len(box.charbox) - 1):
                        # if box.charbox[j].label.find('合')>=0 or box.charbox[j].label.find('格')>=0:
                        #      continue
                        point = box.charbox[j].points
                        x1, y1 = [float(i) for i in point[0]]
                        x2, y2 = [float(i) for i in point[1]]
                        x3, y3 = [float(i) for i in point[2]]
                        x4, y4 = [float(i) for i in point[3]]
                        fw.write(
                            str.format('{0},{1},{2},{3},{4},{5},{6},{7},{8},', x1, y1, x2, y2, x3, y3, x4, y4,
                                       box.charbox[j].label))
                    point = box.charbox[len(box.charbox) - 1].points
                    x1, y1 = [float(i) for i in point[0]]
                    x2, y2 = [float(i) for i in point[1]]
                    x3, y3 = [float(i) for i in point[2]]
                    x4, y4 = [float(i) for i in point[3]]
                    fw.write(str.format('{0},{1},{2},{3},{4},{5},{6},{7},{8}\r\n', x1, y1, x2, y2, x3, y3, x4, y4,
                                        box.charbox[len(box.charbox) - 1].label))
                    # fw.write('\r\n')
            fw.close()
        fr.close()


def cmpfile():
    path = './200/lmr20190516/*.jpg'
    list_jpg = glob.glob(path)
    # print len(list_jpg),len(glob.glob('./100/*.json')),len(glob.glob('./100/*.jpg.txt'))
    for num, jpg_file in enumerate(list_jpg):
        (filepath, tempfilename) = os.path.split(jpg_file)
        (filename, extension) = os.path.splitext(tempfilename)
        if not os.path.exists(os.path.join('./200/lmr20190516/', filename + '.json')):
            print(tempfilename)


# 批量复制文件到指定目录下
def obj2json(obj):
    return {'accuracy_cls': obj.accuracy_cls,
            'eta': obj.eta,
            'iter': obj.iter,
            'loss': obj.loss,
            'loss_bbox': obj.loss_bbox,
            'loss_char_mask': obj.loss_char_mask,
            'loss_cls': obj.loss_cls,
            'loss_global_mask': obj.loss_global_mask,
            'loss_rpn_bbox_fpn2': obj.loss_rpn_bbox_fpn2,
            'loss_rpn_bbox_fpn3': obj.loss_rpn_bbox_fpn3,
            'loss_rpn_bbox_fpn4': obj.loss_rpn_bbox_fpn4,
            'loss_rpn_bbox_fpn5': obj.loss_rpn_bbox_fpn5,
            'loss_rpn_bbox_fpn6': obj.loss_rpn_bbox_fpn6,
            'loss_rpn_cls_fpn2': obj.loss_rpn_cls_fpn2,
            'loss_rpn_cls_fpn3': obj.loss_rpn_cls_fpn3,
            'loss_rpn_cls_fpn4': obj.loss_rpn_cls_fpn4,
            'loss_rpn_cls_fpn5': obj.loss_rpn_cls_fpn5,
            'loss_rpn_cls_fpn6': obj.loss_rpn_cls_fpn6,
            'lr': obj.lr,
            'mb_qsize': obj.mb_qsize,
            'mem': obj.mem,
            'time': obj.time
            }


class logs(object):
    def __init__(self, accuracy_cls, eta, iter, loss, loss_bbox, loss_char_mask, loss_cls, loss_global_mask,
                 loss_rpn_bbox_fpn2, loss_rpn_bbox_fpn3, loss_rpn_bbox_fpn4, loss_rpn_bbox_fpn5,
                 loss_rpn_bbox_fpn6, loss_rpn_cls_fpn2, loss_rpn_cls_fpn3, loss_rpn_cls_fpn4, loss_rpn_cls_fpn5,
                 loss_rpn_cls_fpn6):
        self.accuracy_cls = accuracy_cls
        self.eta = eta
        self.iter = iter
        self.loss = loss
        self.loss_bbox = loss_bbox
        self.loss_char_mask = loss_char_mask
        self.loss_cls = loss_cls
        self.loss_global_mask = loss_global_mask
        self.loss_rpn_bbox_fpn2 = loss_rpn_bbox_fpn2
        self.loss_rpn_bbox_fpn3 = loss_rpn_bbox_fpn3
        self.loss_rpn_bbox_fpn4 = loss_rpn_bbox_fpn4
        self.loss_rpn_bbox_fpn5 = loss_rpn_bbox_fpn5
        self.loss_rpn_bbox_fpn6 = loss_rpn_bbox_fpn6
        self.loss_rpn_cls_fpn2 = loss_rpn_cls_fpn2
        self.loss_rpn_cls_fpn3 = loss_rpn_cls_fpn3
        self.loss_rpn_cls_fpn4 = loss_rpn_cls_fpn4
        self.loss_rpn_cls_fpn5 = loss_rpn_cls_fpn5
        self.loss_rpn_cls_fpn6 = loss_rpn_cls_fpn6

    def __repr__(self):
        return repr((self.accuracy_cls, self.iter, self.loss, self.loss_global_mask, self.loss_char_mask))


# import matplotlib.pyplot as plt


def statistics_model_train():
    total_iters = 199999
    accuracy = []  # = np.zeros(total_iters)
    loss = []  # = np.zeros(total_iters)
    iter = []
    path = '/Volumes/Home/masktext/tools/train/jiashili20190510_train/generalized_rcnn/shrink++/20190522_223237.log'
    with open(path, 'r') as res:
        index = 0
        num = 0
        for line in res.readlines():
            line = line.strip().split('  ')
            if 'json_stats:' in line[1]:
                json_stats = (json.loads(line[1][12:]))
                for i in range(json_stats['iter'] - num):
                    accuracy.append(float(json_stats['accuracy_cls']))
                    loss.append(float(json_stats['loss']))
                    iter.append(float(json_stats['iter']))
                    index = index + 1
                num = int(json_stats['iter'])
                print (index)
                # print line[-38],line[-37],line[-8],line[-7],line[-40],line[-39]
    res.close()
    plt.plot(loss, 'b')
    plt.plot(accuracy, 'r')
    # plt.plot(iter, 'g')
    plt.legend(('Loss', 'Accuracy'), loc='upper right')
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.title('Model training process')
    plt.grid(True)
    plt.show()


def statistics_model_train():
    accuracy = []  # = np.zeros(total_iters)
    loss = []  # = np.zeros(total_iters)
    iter = []
    path = '/Volumes/Home/masktext/tools/train/jiashili20190510_train/generalized_rcnn/shrink++/20190522_223237.log'
    with open(path, 'r') as res:
        index = 0
        num = 0
        for line in res.readlines():
            line = line.strip().split('  ')
            if 'json_stats:' in line[1]:
                json_stats = json.loads(line[1][12:])
                if json_stats['iter'] in [39980, 79980, 119980, 159980, 199999]:
                    print (str.format('accuracy_cls:{0},iter:{1},loss:{2},lr:{3}', json_stats['accuracy_cls'],
                                     json_stats['iter'], json_stats['loss'], json_stats['lr']))
    res.close()


def statistics_score():
    file_path = '/Volumes/Home/masktext/tools/train/jiashili20190510_train/generalized_rcnn/shrink++/jiashili20190510_test/model_final.pkl_results/'
    list_txt = glob.glob(os.path.join(file_path, '*.txt'))
    # img = np.load(file_path+'res_img_2_5.mat.npy')
    score = 0
    index = 0
    for num, txt_file in enumerate(list_txt):
        with open(txt_file, 'r') as fr:
            for line in fr.readlines():
                line = line.strip().split(';')
                line = line[2].split(',')
                score = score + float(line[2])
                index = index + 1
    print (score / index, index)


# def statistics_char_score():
#     recong_gts_path = '/Volumes/Home/masktext/tools/train/jiashili20190510_train/generalized_rcnn/shrink++/jiashili20190510_test/model_final.pkl_results/'
#     test_gts_path = '/Volumes/dataset/jiashili20190510/test_gts/'
#     list_test_gts = glob.glob(os.path.join(test_gts_path, '*.jpg.txt'))
#     list_recong_gts = glob.glob(os.path.join(recong_gts_path, '*.txt'))
#     # img = np.load(file_path+'res_img_2_5.mat.npy')
#     score = 0
#     index = 0
#     for num, txt_file in enumerate(list_txt):
#         with open(txt_file, 'r') as fr:
#             for line in fr.readlines():
#                 line = line.strip().split(';')
#                 line = line[2].split(',')
#
#                 score = score + float(line[2])
#                 index = index + 1
#     print score / index, index


# 标注json文件匹配jpg文件，并把对应的jpg文件拷贝到指定目录
def copytodir():
    source = '/media/wyu/software/JiaShiLiData/jpg/300/'
    target = '/media/wyu/software/new_image/300_json/'
    list_jpg = glob.glob(os.path.join(source, '*.jpg'))
    list_json = glob.glob(os.path.join(target, '*.json'))
    for num, json_file in enumerate(list_json):
        for i, jpg_file in enumerate(list_jpg):
            json_name = os.path.basename(json_file).split('.')[0]
            jpg_name = os.path.basename(jpg_file).split('.')[0]
            if json_name == jpg_name:
                shutil.copyfile(jpg_file, str.format('{0}{1}.jpg', target, jpg_name))


# difference,union,intersection
def file_union():
    A = '/Users/zhoujianwen.cn/Desktop/100/train_100'
    B = '/Users/zhoujianwen.cn/Desktop/100/train_103'
    target = '/Users/zhoujianwen.cn/Desktop/100/'
    list_jpg = glob.glob(os.path.join(A, '*.jpg'))
    list_json = glob.glob(os.path.join(B, '*.json'))
    list_jpg_json = list(set(list_jpg).union(set(list_json)))
    for file in list_jpg_json:
        shutil.copy(file, str.format('{0}{1}', target, os.path.basename(file)))


def file_difference():
    A = './100/'
    B = './data/train/'
    target = './200/'
    list_A = glob.glob(os.path.join(A, '*.jpg'))
    list_B = glob.glob(os.path.join(B, '*.jpg'))
    for num, a in enumerate(list_A):
        list_A[num] = a.split('/')[-1]

    for i, a in enumerate(list_A):
        ajpg = a.split('/')[-1]
        list_A[i] = ajpg
        for j, b in enumerate(list_B):
            bjpg = b.split('/')[-1]
            list_B[j] = bjpg
            if ajpg == bjpg:
                list_B.remove(list_B[j])
    for num, file in enumerate(list_B):
        print  num
        shutil.copy('./data/train/' + file, str.format('{0}{1}', target, file))


import os, base64


def strToImage():
    with open('./100/571.json', 'r') as fr:
        data = json.load(fr)
    fr.close()
    imgdata = base64.b64decode(data['imageData'])
    fw = open('./100/temp.jpg', 'wb')
    fw.write(imgdata)
    fw.close()


def divide():
    # coding=utf-8

    import os
    import os.path
    import shutil
    import random
    import numpy as np

    source_dir = 'F:/cocoDataAugment/data/rerotate_resize/'  # 图片存储的文件夹名称
    # 20%的数据生成验证集
    # val_dir = '/home/jie/Desktop/product_date/masktext/lib/datasets/data/moudle_val/val_img/'
    # 60%的数据生成训练集
    train_dir = 'F:/cocoDataAugment/data/0_0train/'
    # 20%的数据生成训练集
    test_dir = 'F:/cocoDataAugment/data/0_0test/'

    # f1 = file(val_dir + "val_list.txt", "a+")
    f2 = file(train_dir + "train_list.txt", "a+")
    f3 = file(test_dir + "test_list.txt", "a+")

    i = 1
    for root, dirs, files in os.walk(source_dir):
        l = len(files) / 3
        print(files)
        random.shuffle(files)
        print(files)
        for file_single in files:
            if 'txt' not in file_single and 'json' not in file_single:
                if i < int(l * 0.7):
                    # 0-0.6  20%
                    # jpg file
                    # json file
                    shutil.copy(source_dir + file_single, train_dir+'train_imgs')
                    shutil.copy(source_dir + file_single.split('.')[0] + '.json', train_dir+'json')
                    shutil.copy(source_dir + file_single.split('.')[0] + '.jpg.txt', train_dir+'train_gts')
                    f2.write(file_single + '\n')
                    i = i + 1
                # elif i < int(l * 0.8) and i >= math.ceil(l * 0.6):
                #     # 0.6-0.8  20%
                #     # jpg file
                #     # json file
                #     shutil.copy(source_dir + file_single, val_dir)
                #     # shutil.copy(source_dir + file_single.split('.')[0] + '.json', val_dir)
                #     shutil.copy(source_dir + file_single.split('.')[0] + '.jpg.txt', val_dir)
                #     f1.write(file_single + '\n')
                #     i = i + 1
                else:
                    # 0.8-1.0 20%
                    shutil.copy(source_dir + file_single, test_dir+'test_imgs')
                    shutil.copy(source_dir + file_single.split('.')[0] + '.json', test_dir+'json')
                    shutil.copy(source_dir + file_single.split('.')[0] + '.jpg.txt', test_dir+'test_imgs')
                    f3.write(file_single + '\n')
                    i = i + 1
            else:
                continue
    # f1.close()
    f2.close()
    f3.close()

def create_list():

    test_dir = '/home/jie/Desktop/rotated_img/'
    f1 = file(test_dir + "test_list.txt", "a+")
    for root, dirs, files in os.walk(test_dir):
        for file_single in files:
            f1.write(file_single + '\n')
            f2 = file(test_dir + file_single + ".txt", "a+")



if __name__ == '__main__':
    print ('reading......')
    # create_list()
    # json2txt()
    divide()

    # generateData()
    # check_label1()
    # check_label_len(3)

    # find_Img_Singlelabel()
    # single_json2txt()

    # cmpfile()
    # strToImage()
    # statistics_score()
    # statistics_model_train()

    # file_difference()
    # copyfiles_train()
    # copytomasktext()
    # copyfiles_test()
    # copyfileSingleImg()
    # copytodir()
    # file_union()
    # d={'1i1ee':5,'aangy':32,'liqun':62,'lidaming':1}
    #
    # d = sorted(d.items(), key=lambda item: item[1])
    # print d

'''
python 两个list 求交集，并集，差集

listA = [1,3,65,2,7]
listB = [3,2,5,4]

c = [x for x in listA if x in listB]
d = [y for y in (listA+listB) if y not in c]

print(c)
print(d)

def diff(listA,listB):
    #求交集的两种方式
    retA = [i for i in listA if i in listB]
    retB = list(set(listA).intersection(set(listB)))

    print "retA is: ",retA
    print "retB is: ",retB

    #求并集
    retC = list(set(listA).union(set(listB)))
    print "retC1 is: ",retC

    #求差集，在B中但不在A中
    retD = list(set(listB).difference(set(listA)))
    print "retD is: ",retD

    retE = [i for i in listB if i not in listA]
    print "retE is: ",retE
---------------------------------------------------------------

shell统计当前文件夹下的文件个数、目录个数
1、 统计当前文件夹下文件的个数
　　ls -l |grep "^-"|wc -l

2、 统计当前文件夹下目录的个数
　　ls -l |grep "^d"|wc -l

3、统计当前文件夹下文件的个数，包括子文件夹里的 
　　ls -lR|grep "^-"|wc -l

4、统计文件夹下目录的个数，包括子文件夹里的
　　ls -lR|grep "^d"|wc -l

grep "^-" 
　　这里将长列表输出信息过滤一部分，只保留一般文件，如果只保留目录就是 ^d

wc -l 
　　统计输出信息的行数，因为已经过滤得只剩一般文件了，所以统计结果就是一般文件信息的行数，又由于一行信息对应一个文件，所以也就是文件的个数。
'''