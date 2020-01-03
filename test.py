# -*- coding: utf-8 -*-
# author = 'jie'


import os
import io
import json
import re
import cal_IoU as cI
# s = '12.11,1,234,2,13465.465465454,1,2,1'
# a = re.findall(r'(\d+\.\d+),\d|[^.](\d\d+),\d', s)
# # if re.search(r'o\d+o',s):
# #     print('dsfs')
# x = s.split(',',4)
# print(x)
# # print(a)
# exit()
#
label_txt_path = 'D:\MyApplication\github/tools/'
chars_path = 'D:\MyApplication\github/tools/'
def cal_box_acc(test_box):
    true = 0
    label_len = 0
    extra = 0
    miss = 0
    for i in range(len(test_box)):
        label = get_one_box(label_txt_path,test_box[i]['image_name'])
        print('label: ',label)
        print('test_box: ',test_box)
        label_len = label_len + len(label['box_list'])
        if len(label['box_list'])>=len(test_box[i]['box_list']):
            for j in range(len(label['box_list'])):
                flag = 0
                #print(label['box_list'][j])
                for k in range(len(test_box[i]['box_list'])):
                    #print(test_box[i]['box_list'][k])
                    value = cI.cal_IoU(label['box_list'][j], test_box[i]['box_list'][k])
                    if value > 0.8:
                        true = true + 1
                        flag = 1
                        break
                if flag == 0:
                    print(label['image_name'])
            miss = miss+len(label['box_list'])-len(test_box[i]['box_list'])
        else:
            for j in range(len(label['box_list'])):
                flag = 0
                for k in range(len(test_box[i]['box_list'])):
                    value = cI.cal_IoU(label['box_list'][j], test_box[i]['box_list'][k])
                    if value > 0.8:
                        true = true + 1
                        flag = 1
                        break
                if flag == 0:
                    print(label['image_name'])
                # if flag == 1:
                #     break

            extra = extra + len(test_box[i]['box_list']) - len(label['box_list'])
    print(true)
    print(label_len)
    return float(true)/label_len, extra, miss

def get_test_box(path):
    dirs = os.listdir(path)
    files = []
    test_box = []

    for i in dirs:
        if 'char' in i and 'txt' in i:  # 筛选txt文件
            files.append(path+i)

    for file in files: # 遍历文件夹
        # print(file)
        t = {}
        t['box_list'] = []
        t['char'] = []
        t['image_name'] = file.split('char_')[-1].split('.')[0]
        f = open(file)  # 打开文件
        iter_f = iter(f)  # 创建迭代器
        for line in iter_f:  # 遍历文件，一行行遍历，读取文本
            box_coordinate = re.findall(r'\[(.*)\]', line)
            t['char'].append(re.findall(r'([0-9o])_',line)[0])
            for o in box_coordinate:
                one_coordinate = o.split(',')
                one_coordinate = [float(x) for x in one_coordinate]
                t['box_list'].append(one_coordinate)
        test_box.append(t)
    return test_box

def get_one_box(label_txt_path,name):
    label = {}
    label['image_name'] = name
    f = open(label_txt_path+name+'.jpg.txt')  # 打开文件
    iter_f = iter(f)  # 创建迭代器
    label['box_list'] = []
    label['char'] = []
    for line in iter_f:  # 遍历文件，一行行遍历，读取文本
        box_coordinate = line.split(',')
        one_coordinate = []
        count = 0
        for o in range(len(box_coordinate)):
            if o+count>=len(box_coordinate):
                break
            if re.search(r'o\d+o',box_coordinate[o+8]):
                # print(box_coordinate[o+8])
                count = count+9
            if (o+1) % 9 == 0 and o != 0:
                # print(box_coordinate[o])
                label['char'].append(re.sub('\n','',box_coordinate[o+count]))
                label['box_list'].append(one_coordinate)
                one_coordinate = []
            else:
                one_coordinate.append(float(box_coordinate[o+count]))
    return label

t_box = get_test_box(chars_path)
# t_box = get_test_box('/home/jie/Desktop/product_date/masktext/results/train/rotate_train/moudle_8_test/model_final.pkl_results/test/')

print(cal_box_acc(t_box))
# box_coordinate = re.findall(r'(\f,\f,\f,\f),\d|(\f,\f,\f,\f),o', line)
# if len(box_coordinate) < 1:
#     print(name)
#     return label
# for o in box_coordinate:
#     one_coordinate = o.split(',')
#     one_coordinate = [float(x) for x in one_coordinate]
#     label['box_list'].append(one_coordinate)