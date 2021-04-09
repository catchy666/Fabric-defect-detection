import cv2 as cv
import json
import os
from tqdm import tqdm

"""
Crop Dataset Images
"""

# pre-settings
SIZE = 1024  # size of crop image
ROOT_PATH = '../tcdata/test/'  # dataset root path
IMG_PATH = ROOT_PATH + 'defect'  # raw images directory
IMG_ANN = ROOT_PATH + 'annotations/ann_test.json'  # raw annotations info

CROP_IMG_PATH = ROOT_PATH + 'defect_{}'.format(SIZE)  # crop images save path
CROP_IMG_ANN = ROOT_PATH + 'annotations/ann_test_{}.json'.format(SIZE)  # crop annotations info


class Crop:
    """
    Crop origin image to slices.
    """
    def __init__(self, crop_path, crop_size, img_anns):
        self.crop_path = crop_path  # slice
        self.crop_size = crop_size  # slice size
        self.img_anns = img_anns  # raw annotations file
        self.new_img_anns = []  # crop images annotations

    def get_json(self):
        """
        Return annotations info.
        """
        new_anns = {}
        img_anns = json.load(open(self.img_anns, 'r', encoding="UTF-8"))

        for ann in img_anns:
            # each fabric defect info
            defect = {}
            defect['defect_name'] = ann['defect_name']  # defect name
            defect['bbox'] = ann['bbox']  # defect bounding box (xy, xy)

            if ann['name'] in new_anns.keys():
                new_anns[ann['name']].append(defect)
            else:
                new_anns[ann['name']] = [defect]
        return new_anns

    def crop_image(self, img_dir, step):
        """
        crop image + return each slice annotation
        """
        defects_info = self.get_json() # dict{ 'image-name.jpg': [dict{'defect_name': defect_name, 'bbox': xyxy}]}

        for img_name in tqdm(os.listdir(img_dir)):
            img = cv.imread(os.path.join(img_dir, img_name)) # read image
            image_size = img.shape  # origin image size
            xl = 0  # 左上角点的坐标
            yl = 0  # 左上角点的坐标
            position_x = 0  # x轴裁剪到第几张图
            position_y = 0  # y轴
            crop_info = {}

            # cropping
            while yl + position_y * step + self.crop_size < image_size[1]:
                while xl + position_x * step + self.crop_size < image_size[0]:
                    # slice info
                    aftercrop_img = {}
                    aftercrop_img['image'] = img[xl + position_x * step:xl + position_x * step + self.crop_size,
                                             yl + position_y * step:yl + position_y * step + self.crop_size]
                    aftercrop_img['position_x'] = position_x
                    aftercrop_img['position_y'] = position_y
                    aftercrop_img['image_height'] = self.crop_size
                    aftercrop_img['image_width'] = self.crop_size
                    crop_info[
                        img_name.split('.')[0] + '_' + str(position_x) + '_' + str(position_y) + '.jpg'] = aftercrop_img

                    position_x += 1
                aftercrop_img = {}

                aftercrop_img['image'] = img[xl + position_x * step:image_size[0],
                                         yl + position_y * step:yl + position_y * step + self.crop_size]
                aftercrop_img['position_x'] = position_x
                aftercrop_img['position_y'] = position_y
                aftercrop_img['image_height'] = self.crop_size
                aftercrop_img['image_width'] = image_size[0] - (xl + position_x * step)
                crop_info[
                    img_name.split('.')[0] + '_' + str(position_x) + '_' + str(position_y) + '.jpg'] = aftercrop_img
                position_y += 1
                position_x = 0
            while xl + position_x * step + self.crop_size < image_size[0]:
                aftercrop_img = {}
                aftercrop_img['image'] = img[xl + position_x * step:xl + position_x * step + self.crop_size,
                                         yl + position_y * step:image_size[1]]
                aftercrop_img['position_x'] = position_x
                aftercrop_img['position_y'] = position_y
                aftercrop_img['image_height'] = image_size[1] - (yl + position_y * step)
                aftercrop_img['image_width'] = self.crop_size
                crop_info[
                    img_name.split('.')[0] + '_' + str(position_x) + '_' + str(position_y) + '.jpg'] = aftercrop_img
                position_x += 1

            aftercrop_img = {}
            aftercrop_img['image'] = img[xl + position_x * step:image_size[0], yl + position_y * step:image_size[1]]
            aftercrop_img['position_x'] = position_x
            aftercrop_img['position_y'] = position_y
            aftercrop_img['image_height'] = image_size[1] - (yl + position_y * step)
            aftercrop_img['image_width'] = image_size[0] - (xl + position_x * step)
            crop_info[img_name.split('.')[0] + '_' + str(position_x) + '_' + str(position_y) + '.jpg'] = aftercrop_img

            # After cropped this image, get each slice's defect info:
            name_info = []
            image_info = img_name.split('.')[0] + '.jpg'

            for bad_infor in defects_info[image_info]:
                new_img_ann = {}  # 每张瑕疵点所在切图中的信息
                bbox = bad_infor['bbox']  # 瑕疵原来的框
                image_name = image_info.split('.')[0]
                x_min = bbox[0]
                y_min = bbox[1]
                x_max = bbox[2]
                y_max = bbox[3]
                x_i_min = x_min // (self.crop_size * 0.8)
                y_i_min = y_min // (self.crop_size * 0.8)
                new_x_min = x_min % (self.crop_size * 0.8)
                new_y_min = y_min % (self.crop_size * 0.8)
                x_i_max = x_max // (self.crop_size * 0.8)
                y_i_max = y_max // (self.crop_size * 0.8)
                new_x_max = x_max % (self.crop_size * 0.8)
                new_y_max = y_max % (self.crop_size * 0.8)

                if x_i_min > position_y:
                    x_i_min = position_y
                    x_i_max = position_y
                    new_x_min = x_min - position_y * self.crop_size * 0.8
                    new_x_max = x_max - position_y * self.crop_size * 0.8
                if x_i_max > position_y:
                    x_i_max = position_y
                    new_x_max = x_max - position_y * self.crop_size * 0.8
                if y_i_min > position_x:
                    y_i_min = position_x
                    y_i_max = position_x
                    new_y_min = y_min - position_x * self.crop_size * 0.8
                    new_y_max = y_max - position_x * self.crop_size * 0.8
                if y_i_max > position_x:
                    y_i_max = position_x
                    new_y_max = y_max - position_x * self.crop_size * 0.8

                if x_i_min == x_i_max and y_i_min == y_i_max:
                    new_bbox = [new_x_min, new_y_min, new_x_max, new_y_max]
                    x_i_2 = x_i_min
                    y_i_2 = y_i_min
                elif x_i_min == x_i_max and y_i_min != y_i_max:
                    if self.crop_size - new_y_min > new_y_max:
                        if y_i_min * step + self.crop_size > y_max:
                            new_bbox = [new_x_min, new_y_min, new_x_max, y_max - y_i_min * step]
                        else:
                            new_bbox = [new_x_min, new_y_min, new_x_max, self.crop_size]
                        x_i_2 = x_i_min
                        y_i_2 = y_i_min
                    else:
                        if y_i_max * step > y_min:
                            new_bbox = [new_x_min, 0, new_x_max, new_y_max]
                        else:
                            new_bbox = [new_x_min, y_min - y_i_max * step, new_x_max, new_y_max]
                        x_i_2 = x_i_min
                        y_i_2 = y_i_max
                elif y_i_min == y_i_max and x_i_min != x_i_max:
                    if self.crop_size - new_x_min > new_x_max:
                        if x_i_min * step + self.crop_size > x_max:
                            new_bbox = [new_x_min, new_y_min, x_max - x_i_min * step, new_y_max]
                        else:
                            new_bbox = [new_x_min, new_y_min, self.crop_size, new_y_max]
                        x_i_2 = x_i_min
                        y_i_2 = y_i_min
                    else:
                        if x_i_max * step > x_min:
                            new_bbox = [0, new_y_min, new_x_max, new_y_max]
                        else:
                            new_bbox = [x_min - x_i_max * step, new_y_min, new_x_max, new_y_max]
                        x_i_2 = x_i_max
                        y_i_2 = y_i_min
                else:
                    if self.crop_size - new_x_min > new_x_max and self.crop_size - new_y_min > new_y_max:
                        if x_i_min * step + self.crop_size > x_max and y_i_min * step + self.crop_size > y_max:
                            new_bbox = [new_x_min, new_y_min, x_max - x_i_min * step, y_max - y_i_min * step]
                        elif x_i_min * step + self.crop_size > x_max:
                            new_bbox = [new_x_min, new_y_min, x_max - x_i_min * step, self.crop_size]
                        elif y_i_min * step + self.crop_size > y_max:
                            new_bbox = [new_x_min, new_y_min, self.crop_size, y_max - y_i_min * step]
                        else:
                            new_bbox = [new_x_min, new_y_min, self.crop_size, self.crop_size]
                        x_i_2 = x_i_min
                        y_i_2 = y_i_min
                    elif self.crop_size - new_y_min > new_y_max:
                        if y_i_min * step + self.crop_size > y_max and x_i_max * step > x_min:
                            new_bbox = [0, new_y_min, new_x_max, y_max - y_i_min * step]
                        elif y_i_min * step + self.crop_size > y_max:
                            new_bbox = [x_min - x_i_max * step, new_y_min, new_x_max, y_max - y_i_min * step]
                        elif x_i_max * step > x_min:
                            new_bbox = [0, new_y_min, new_x_max, self.crop_size]
                        else:
                            new_bbox = [0, new_y_min, new_x_max, self.crop_size]
                        x_i_2 = x_i_max
                        y_i_2 = y_i_min
                    elif self.crop_size - new_x_min > new_x_max:
                        if x_i_min * step + self.crop_size > x_max and y_i_max * step > y_min:
                            new_bbox = [new_x_min, 0, x_max - x_i_min * step, new_y_max]
                        elif x_i_min * step + self.crop_size > x_max:
                            new_bbox = [new_x_min, y_min - y_i_max * step, x_max - x_i_min * step, new_y_max]
                        elif y_i_max * step > y_min:
                            new_bbox = [new_x_min, 0, self.crop_size, new_y_max]
                        else:
                            new_bbox = [new_x_min, 0, self.crop_size, new_y_max]
                        x_i_2 = x_i_min
                        y_i_2 = y_i_max
                    else:
                        if x_i_max * step > x_min and y_i_max * step > y_min:
                            new_bbox = [0, 0, new_x_max, new_y_max]
                        elif x_i_max * step > x_min:
                            new_bbox = [0, y_min - y_i_max * step, new_x_max, new_y_max]
                        elif y_i_max * step > y_min:
                            new_bbox = [x_min - x_i_max * step, 0, new_x_max, new_y_max]
                        else:
                            new_bbox = [x_min - x_i_max * step, y_min - y_i_max * step, new_x_max, new_y_max]
                        x_i_2 = x_i_max
                        y_i_2 = y_i_max

                image_num = image_name + '_' + str(int(y_i_2)) + '_' + str(int(x_i_2)) + ".jpg"
                new_img_ann["name"] = image_num
                if image_num not in name_info:
                    name_info.append(image_num)
                new_img_ann["category"] = bad_infor['category']
                new_img_ann["bbox"] = new_bbox
                self.new_img_anns.append(new_img_ann)

            for image_name in name_info:
                cv.imwrite(os.path.join(self.crop_path, image_name), crop_info[image_name]["image"])

        return self.new_img_anns


def main():
    crop = Crop(CROP_IMG_PATH, SIZE, IMG_ANN)
    crop_ann = crop.crop_image(IMG_PATH, int(0.8 * SIZE))  # step = slice_size * 0.8

    with open(CROP_IMG_ANN, 'w') as f:
        json.dump(crop_ann, f, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    main()
