import base64
import json
from labelme import utils
import cv2 as cv
import sys
import numpy as np
import random
import re
import os
import io

source_path = 'F:/cocoDataAugment/data/rotate/'
# source_path = 'F:/cocoDataAugment/data/temp/'
destination_path = 'F:/cocoDataAugment/data/'
# destination_path = 'F:/cocoDataAugment/data/temp/'
gaussian_path = destination_path + 'rotate_gaussian/'
IncreaseEP_path = destination_path + 'rotate_IncreaseEP/'
ReduceEP_path = destination_path + 'rotate_ReduceEP/'
salt_path = destination_path + 'rotate_salt/'
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                L.append(os.path.join(root, file))
        return L

class DataAugment(object):
    def __init__(self, image_id=700):
        self.add_saltNoise = True
        self.gaussianBlur = True
        self.changeExposure = True
        self.id = image_id
        self.img_save_path = destination_path + str(self.id)
        self.gaussian_save_path = gaussian_path + str(self.id)
        self.IncreaseEP_save_path = IncreaseEP_path + str(self.id)
        self.ReduceEP_save_path = ReduceEP_path +  str(self.id)
        self.salt_save_path = salt_path + str(self.id)
        img = cv.imread(source_path + str(self.id)+'.jpg')
        try:
            img.shape
        except:
            print('No Such image!---'+str(id)+'.jpg')
            sys.exit(0)
        self.src = img
        # dst3 = cv.flip(img, -1, dst=None)
        # self.flip_x_y = dst3
        # cv.imwrite(str(self.id)+'_flip_x_y'+'.jpg', self.flip_x_y)

    def gaussian_blur_fun(self):
        if self.gaussianBlur:
            x = random.choice([5, 7])
            dst1 = cv.GaussianBlur(self.src, (x, x), 0)
            # dst4 = cv.GaussianBlur(self.flip_x_y, (5, 5), 0)
            cv.imwrite(self.gaussian_save_path + '_Gaussian'+'.jpg', dst1)
            # cv.imwrite(str(self.id)+'_flip_x_y'+'_Gaussian'+'.jpg', dst4)

    def change_exposure_fun(self):
        if self.changeExposure:
            # contrast
            reduce = 0.8 #越小越暗
            increase = 1.2 # 1.4 越大越亮
            # brightness
            g = 10
            h, w, ch = self.src.shape
            add = np.zeros([h, w, ch], self.src.dtype)
            dst1 = cv.addWeighted(self.src, reduce, add, 1-reduce, g)
            dst2 = cv.addWeighted(self.src, increase, add, 1-increase, g)
            # dst7 = cv.addWeighted(self.flip_x_y, reduce, add, 1 - reduce, g)
            # dst8 = cv.addWeighted(self.flip_x_y, increase, add, 1 - increase, g)
            cv.imwrite(self.ReduceEP_save_path + '_ReduceEp' + '.jpg', dst1)
            # cv.imwrite(str(self.id)+'_flip_x_y'+'_ReduceEp'+'.jpg', dst7)
            cv.imwrite(self.IncreaseEP_save_path+'_IncreaseEp'+'.jpg', dst2)
            # cv.imwrite(str(self.id)+'_flip_x_y'+'_IncreaseEp'+'.jpg', dst8)

    def add_salt_noise(self):
        if self.add_saltNoise:
            # percentage = 0.005
            # percentage = random.choice([0.02, 0.01, 0.03, 0.04, 0.05, 0.06])
            percentage = 0.01
            dst1 = self.src
            # dst4 = self.flip_x_y
            num = int(percentage * self.src.shape[0] * self.src.shape[1])
            for i in range(num):
                rand_x = random.randint(0, self.src.shape[0] - 1)
                rand_y = random.randint(0, self.src.shape[1] - 1)
                if random.randint(0, 1) == 0:
                    dst1[rand_x, rand_y] = 0
                    # dst4[rand_x, rand_y] = 0
                else:
                    dst1[rand_x, rand_y] = 255
                    # dst4[rand_x, rand_y] = 255
            cv.imwrite(self.salt_save_path+'_Salt'+'.jpg', dst1)
            # cv.imwrite(str(self.id)+'_flip_x_y'+'_Salt'+'.jpg', dst4)

    def json_generation(self):
        # image_names = [str(self.id)+'_flip_x', str(self.id)+'_flip_y', str(self.id)+'_flip_x_y']
        image_names = []
        if self.gaussianBlur:
            image_names.append(self.gaussian_save_path+'_Gaussian')
            # image_names.append(str(self.id)+'_flip_x_y'+'_Gaussian')
        if self.changeExposure:
            image_names.append(self.ReduceEP_save_path+'_ReduceEp')
            # image_names.append(str(self.id)+'_flip_x_y'+'_ReduceEp')
            image_names.append(self.IncreaseEP_save_path+'_IncreaseEp')
            # image_names.append(str(self.id)+'_flip_x_y'+'_IncreaseEp')
        if self.add_saltNoise:
            image_names.append(self.salt_save_path+'_Salt')
            # image_names.append(str(self.id)+'_flip_x_y' + '_Salt')

        with open(source_path + str(self.id) + ".json", 'r')as js:
            json_data = json.load(js)
            for image_name in image_names:
                # img = utils.img_b64_to_arr(json_data['imageData'])
                height, width = json_data['imageHeight'], json_data['imageWidth']
                shapes = json_data['shapes']
                for shape in shapes:
                    points = shape['points']
                    for point in points:
                        # match_pattern4 = re.compile(r'(.*)_x_y(.*)')
                        # if match_pattern4.match(image_name):
                        #     point[0] = width - point[0]
                        #     point[1] = height - point[1]
                        # else:
                        point[0] = point[0]
                        point[1] = point[1]
                json_data['imagePath'] = image_name.split('/')[-1]+".jpg"
                json_data['imageData'] = None
                json.dump(json_data, open(image_name+".json", 'w'), indent=2)


if __name__ == "__main__":
    i = 0
    l = len(file_name(source_path))
    # salt 0.05 0.03 0.01 gaussian 3 5 7 9
    # salt 0.05 0.03
    for name in enumerate(file_name(source_path)):
        shape_json = []
        m_path = name[1]
        dir = os.path.dirname(m_path)
        print(os.path.splitext(m_path)[0].split('/')[-1], 'process:', float(i+1)/l)
        i = i + 1
        dataAugmentObject = DataAugment(os.path.splitext(m_path)[0].split('/')[-1])
        dataAugmentObject.changeExposure = False
        # dataAugmentObject.add_saltNoise = False
        dataAugmentObject.gaussianBlur = False
        # dataAugmentObject.gaussian_blur_fun()
        # dataAugmentObject.change_exposure_fun()
        dataAugmentObject.add_salt_noise()
        dataAugmentObject.json_generation()
