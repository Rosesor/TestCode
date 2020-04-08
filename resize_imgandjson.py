# -*- coding: utf-8 -*-
import os
import json
import io
import cv2
import numpy as np
from PIL import Image


source_path = 'F:/cocoDataAugment/data/rotate_salt0.05/'
destination_path = 'F:/cocoDataAugment/data/rotate_salt_5_resize/'
# source_path = 'F:/cocoDataAugment/data/'
# destination_path = 'F:/cocoDataAugment/data/'
new_width = 1280
new_height = 1024

article_info = {}
data_json = json.loads(json.dumps(article_info,indent=4))
print(data_json)
data_json['version'] = '3.16.7'
data_json['flags'] = {}
data_json["lineColor"] = [
    0,
    255,
    0,
    128
]
data_json["fillColor"] = [
    255,
    0,
    0,
    128
]


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                L.append(os.path.join(root, file))
        return L

def resize_img(img,new_width,new_height):
    # 保持长宽比缩放
    # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # img = cv2.resize(img, (new_width,new_height),interpolation = cv2.INTER_AREA)
    # return img
    old_size = img.shape[0:2]
    target_size = [new_height, new_width]
    # ratio = min(float(target_size)/(old_size))
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i * ratio) for i in old_size])
    # new_size=(高，宽)
    img = cv2.resize(img, (new_size[1], new_size[0]))
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    return img_new, new_size[1], new_size[0], pad_w, pad_h, img.shape[1], img.shape[0], old_size

def resize_point(old_w, old_h, actual_w, actual_h,pad_width, pad_height, point):
    # 按比例缩放json的点
    x_rate = float(old_w)/float(actual_w)
    y_rate = float(old_h)/float(actual_h)
    # 将point变成两列（x，y)
    point = np.reshape(point, newshape=(int(len(point) / 2), 2))
    # point旋转
    for i in range(0,len(point)):
        point[i][0]=float(point[i][0])/x_rate + float(pad_width)/2
        point[i][1]=float(point[i][1])/y_rate + float(pad_height)/2
    # point变回一列
    point = np.reshape(point, newshape=(int(len(point) / 4), 8))
    return point


for name in enumerate(file_name(source_path)):
    shape_json = []
    m_path = name[1]
    dir = os.path.dirname(m_path)
    file_json = io.open(m_path, 'r', encoding='utf-8')
    json_data = file_json.read()
    data = json.loads(json_data)
    data_json['imageData'] = None
    data_name = data['imagePath']
    data_path = dir + '/' + data_name
    # angle_item = 45
    object_name = os.path.splitext(data['imagePath'])[0]
    img = cv2.imread(data_path)
    im_resize, resize_w, resize_h, pad_w, pad_h, img_w, img_h, old_size = resize_img(img,1280,1024)
    (filename, extension) = os.path.splitext(data_name)
    data_new_picture_name = destination_path + "/" + filename + '_5' + "_resize" + ".jpg"
    data_new_json_name = destination_path + "/" + filename + "_resize" + ".json"
    data_json['imagePath'] = filename + "_resize" + ".jpg"
    print(filename)
    cv2.imwrite(data_new_picture_name, im_resize)
    data_json['imageWidth'] = new_width
    data_json['imageHeight'] = new_height

    for i in range(len(data['shapes'])):
        point = np.array([])
        assert len(data['shapes'][i]['points']) == 4, object_name+'.jpg has more than 4 points'
        for j in range(0,4):
            point= np.append(point, data['shapes'][i]['points'][j][0])
            point= np.append(point, data['shapes'][i]['points'][j][1])
        m_name_0 = data['shapes'][i]['label']
        data_json_line_color = data['shapes'][i]['line_color']
        data_json_fill_color = data['shapes'][i]['fill_color']
        data_json_rec = data['shapes'][i]['shape_type']
        point = resize_point(old_size[1], old_size[0], img_w, img_h, pad_w, pad_h, point)
        shape_json_item = {"label": m_name_0,
                           "line_color": data_json_line_color,
                           "fill_color": data_json_fill_color,
                           "points": [[point[0][0], point[0][1]],
                                      [point[0][2], point[0][3]],
                                      [point[0][4], point[0][5]],
                                      [point[0][6], point[0][7]]],
                           "shape_type": data_json_rec}
        shape_json.append(shape_json_item)
    data_json['shapes'] = shape_json
    data_info = json.dumps(data_json, ensure_ascii=False,indent=2)
    fp = open(data_new_json_name, "w+")
    fp.write(data_info)
    fp.close()