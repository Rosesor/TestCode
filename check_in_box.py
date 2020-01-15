#coding=utf-8
import numpy
from shapely.geometry import Polygon,MultiPoint,mapping
# box = [2, 0, 0, 1, 0, 0, 2, 1]
# p1 = [1, 0.5]
# p2 = [2, 3]
def check_in_box(o_point,box):
    if box.size==8:
        box = numpy.asarray(box).reshape(4, 2)
    else:
        print('box len is not 8,:', box)
        exit()
    box = mapping(Polygon(box).convex_hull)
    x = []
    for i in range(0,4):
        x.append(numpy.array(box['coordinates'][0][i], float))
    box = x
    # print(box)
    a = (box[1][0]-box[0][0])*(o_point[1]-box[0][1])-(box[1][1]-box[0][1])*(o_point[0]-box[0][0]) #(B.x - A.x)*(y - A.y) - (B.y - A.y)*(x - A.x)
    b = (box[2][0]-box[1][0])*(o_point[1]-box[1][1])-(box[2][1]-box[1][1])*(o_point[0]-box[1][0])
    c = (box[3][0]-box[2][0])*(o_point[1]-box[2][1])-(box[3][1]-box[2][1])*(o_point[0]-box[2][0])
    d = (box[0][0]-box[3][0])*(o_point[1]-box[3][1])-(box[0][1]-box[3][1])*(o_point[0]-box[3][0])
    # print(a,b,c,d)
    if (a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0 and c < 0 and d < 0):
        return True
    elif a==0 or b==0 or c == 0 or d == 0:
        print(o_point,box)
    else:
        return False
# check_in_box(p2,box)