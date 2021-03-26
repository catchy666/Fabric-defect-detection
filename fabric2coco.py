import os
import json
import mmcv
import numpy as np
from tqdm import tqdm
from icecream import ic

defect_name2label = {
    '沾污': 1, '错花': 2, '水印': 3, '花毛': 4, '缝头': 5, '缝头印': 6, '虫粘': 7, '破洞': 8, '褶子': 9,
    '织疵': 10, '漏印': 11, '蜡斑': 12, '色差': 13, '网折': 14, '其他': 15
}


class Fabric2COCO:
    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def to_coco(self, ann_file, out_file, image_prefix):
        # step1: initialize categories
        self._init_categories()

        # step2: editing keys: 'annotations', 'images'
        ann_list = mmcv.load(ann_file)  # type(ann_file) = list(dict()); len(ann_list) = 21323
        name_list = list(set([_['name'] for _ in ann_list]))  # len(name_list) = 4371
        for img_name in tqdm(name_list):
            img_ann = [_ for _ in ann_list if _['name'] == img_name]
            if len(img_ann) > 100:
                print('{}: >100 defect object.'.format(img_name))
                continue

            # image info dict
            img_path = os.path.join(image_prefix, img_name)
            height, width = mmcv.imread(img_path).shape[:2]
            self.images.append(dict(id=self.img_id,
                                    file_name=os.path.basename(img_path),
                                    height=height,
                                    width=width))

            # annotation info dict
            for info in img_ann:
                bbox = info['bbox']
                if bbox[1] >= height or bbox[0] >= width:
                    continue
                label = defect_name2label[info['defect_name']]  # str -> int label
                annotation = self._annotation(label, bbox, height, width)
                self.annotations.append(annotation)
                self.ann_id += 1

            self.img_id += 1

        # step3: summary
        instance = {}
        instance['info'] = 'fabric defect'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        with open(out_file, 'w') as fp:
            json.dump(instance, fp, indent=4, separators=(',', ': '))

    def _init_categories(self):
        for k, v in defect_name2label.items():
            category = {}
            category['id'] = v
            category['name'] = k
            category['supercategory'] = 'defect_name'
            self.categories.append(category)

    def _annotation(self, label, bbox, h, w):
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area <= 0:
            print(bbox)
            input()
        points = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = label
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points, h, w)
        annotation['iscrowd'] = 0
        annotation['area'] = area
        return annotation

    def _get_box(self, points, img_h, img_w):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        '''coco,[x,y,w,h]'''
        w = max_x - min_x
        h = max_y - min_y
        if w > img_w:
            w = img_w
        if h > img_h:
            h = img_h
        return [min_x, min_y, w, h]


def f2coco(mode='train'):
    """
    Convert annotations to coco format
    :param mode: 模式选择。total为转换全部annos；train/test为转换各自数据集的annos
    """
    # create
    fabric2coco = Fabric2COCO()

    if mode == 'total':
        img_prefix = './tcdata/imgs_defect'
        ann_file = os.path.join('./tcdata', 'annotations/ann.json')
        out_file = os.path.join('./tcdata', 'annotations/instance_{}.json'.format('total'))
        fabric2coco.to_coco(ann_file, out_file, img_prefix)
    elif mode == 'train' or mode == 'test':
        img_prefix = './tcdata/{}/defect'.format(mode)
        ann_file = os.path.join('./tcdata/{}'.format(mode), 'annotations/ann_{}.json'.format(mode))
        out_file = os.path.join('./tcdata/{}'.format(mode), 'annotations/instance_{}.json'.format(mode))
        fabric2coco.to_coco(ann_file, out_file, img_prefix)
    else:
        ic("Your input 'mode' is WRONG!")


if __name__ == '__main__':
    f2coco(mode='test')
