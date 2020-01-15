#coding=utf-8
import numpy as np
import math
# box = [0, 0, 0, 10, 3, 0, 3, 10]
# p0 = [1, 1.5]
# # p1 = [0, 0]
# # p2 = [0, 1]

def get_distance(po, p1, p2):
    p0 = []
    p0.append(po[0][0])
    p0.append(po[1][0])
    # print('start')
    # print(p0, p0[0], p0[1])
    # print(p1, p1[0], p1[1])
    # print(p2, p2[0], p2[1])
    array_longi  = np.array([p2[0]-p1[0], p2[1]-p1[1]]) #长边
    array_trans = np.array([p2[0]-p0[0], p2[1]-p0[1]])
    # 用向量计算点到直线距离
    array_temp = (float(array_trans.dot(array_longi)) / array_longi.dot(array_longi))   # 注意转成浮点数运算
    array_temp = array_longi.dot(array_temp)
    distance   = np.sqrt((array_trans - array_temp).dot(array_trans - array_temp))
    return distance

def get_long_line(p1, p2):
    if p1[0] > p2[0]:
        p1, p2 = p2, p1
    if p2[0] == p1[0]:
        if p2[1] < p1[1]:
            p1, p2 = p2, p1
    return p1, p2


# degrees = math.degrees(math.atan2(-1 + 0, 0 - 0))
# print(degrees)
def decide_up_or_down(box, p0):
    left = None
    up = None
    if box.size == 8:
        box = np.asarray(box).reshape(4, 2)
    else:
        print('box len is not 8,:', box)
        exit()
    # 获得矩形某一条长边的两点坐标，box_left_bottom、right_bottom
    box_left_top = box[0]
    sort = sort = np.argsort(np.sum(np.asarray(box_left_top - box) ** 2, axis=1))
    box_right_top = box[sort[2]]
    #若两点横坐标相同，则left在下，right在上 ,找的没问题
    box_left_top, box_right_top = get_long_line(box_left_top, box_right_top)
    box_left_bottom = box[sort[1]]
    box_right_bottom = box[sort[3]]
    box_left_bottom, box_right_bottom = get_long_line(box_left_bottom, box_right_bottom)

    dis1 = get_distance(p0, box_left_top, box_right_top)
    dis2 = get_distance(p0, box_left_bottom, box_right_bottom)

    # 计算o点到两边距离
    if dis1<dis2:
        near_line = [box_left_top, box_right_top]
        another_line = [box_left_bottom, box_right_bottom]
    elif dis1==dis2:
        print('o点到两条边的距离一样')
        exit()
    else:
        near_line = [box_left_bottom, box_right_bottom]
        another_line = [box_left_top, box_right_top]

    # print(near_line,another_line)
    # 判断o点的上、下或者左、右位置
    if p0[0] >= near_line[0][0] and p0[0] <= near_line[1][0]: # o点在长边中间，用y轴计算上下
        if float(near_line[0][1]+near_line[1][1])/2 < float(another_line[0][1]+another_line[1][1])/2:
            up = True
            print('上')
        elif float(near_line[0][1]+near_line[1][1])/2 > float(another_line[0][1]+another_line[1][1])/2:
            up = False
            print('下')
        else:
            print('上下两条边重合')
            exit()
    else: #o点上下没有长边，用左右判断
        if float(near_line[0][0]+near_line[1][0])/2 < float(another_line[0][0]+another_line[1][0])/2:
            left = True
            print('left')
        elif float(near_line[0][0]+near_line[1][0])/2 > float(another_line[0][0]+another_line[1][0])/2:
            left = False
            print('right')
        else:
            print('左右两条边重合')
            print('近边，远边',near_line,another_line)
            exit()
    return up, left


