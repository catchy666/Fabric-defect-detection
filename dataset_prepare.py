import mmcv

import numpy as np
import os
import shutil
import glob
import json
from tqdm import tqdm
from icecream import ic

"""
raw_dataset 目录结构如下：
    guangdong1_round2_train2_20191004_Annotations/
    guangdong1_round2_train2_20191004_images/
    guangdong1_round2_train_part1_20190924/
    guangdong1_round2_train_part2_20190924/
    guangdong1_round2_train_part3_20190924/

tcdata 目录结构如下：
    annotations/
    imgs_defect/
    imgs_normal/
"""

RAW_DIR = './raw_dataset/'  # 原始数据集（from 分卷解压缩后）
TO_DIR = './tcdata/'  # 训练所用数据集路径


def copy_annotations():
    """
    将两部分Annotation文件拷贝至 to_dir (tcdata) 目录下
    """
    print("Step 1 is working ...")

    # anno part1:
    raw_ann1 = os.path.join(RAW_DIR, 'guangdong1_round2_train2_20191004_Annotations/Annotations/anno_train.json')
    to_ann1 = os.path.join(TO_DIR, 'annotations/ann_part1.json')
    shutil.copyfile(raw_ann1, to_ann1)
    # anno part2:
    raw_ann2 = os.path.join(RAW_DIR, 'guangdong1_round2_train_part1_20190924/Annotations/anno_train.json')
    to_ann2 = os.path.join(TO_DIR, 'annotations/ann_part2.json')
    shutil.copyfile(raw_ann2, to_ann2)

    print("Step 1 is finished!")


def move_defect_imgs():
    """
    将所有瑕疵图片移动至imgs_defect目录下 | 【4371 defect imgs】
    """
    print("Step 3 is working ...")

    # 获取全部瑕疵图像路径
    dir_list = [_ for _ in os.listdir(RAW_DIR) if 'README.md' != _ and 'Annotations' not in _]

    defect_imgs = []  # path list for all defect images
    for d in dir_list:
        imgs_list = glob.glob(os.path.join(RAW_DIR + d, 'defect') + '/*/*.jpg')
        defect_img = [_ for _ in imgs_list if 'template' not in _]
        defect_imgs += defect_img

    # start moving
    for img in tqdm(defect_imgs):
        shutil.move(img, os.path.join(TO_DIR, 'imgs_defect'))

    print("Step 3 is finished!")


def move_normal_imgs():
    """
    将所有正常（无瑕疵）图片移动至imgs_normal目录下 | 【4371 defect imgs】
    """
    print("Step 4 is working ...")

    # 获取全部正常图像路径
    dir_list = ['guangdong1_round2_train_part2_20190924', 'guangdong1_round2_train2_20191004_images']

    normal_imgs = []  # path list for all normal images
    for d in dir_list:
        imgs_list = glob.glob(os.path.join(RAW_DIR + d, 'normal') + '/*/*.jpg')
        normal_img = [_ for _ in imgs_list if 'template' not in _]
        normal_imgs += normal_img

    ic(len(normal_imgs))
    # start moving
    for img in tqdm(normal_imgs):
        shutil.move(img, os.path.join(TO_DIR, 'imgs_normal'))

    print("Step 4 is finished!")


def merge_annotations():
    """
    merge two annotations json files.
    """
    print("Step 2 is working ...")

    ann1 = mmcv.load(os.path.join(TO_DIR, 'annotations/ann_part1.json'))
    ann2 = mmcv.load(os.path.join(TO_DIR, 'annotations/ann_part2.json'))

    ann1.extend(ann2)
    # mmcv.dump(ann1, os.path.join(TO_DIR, 'annotations/ann.json'), file_format='json')
    with open(os.path.join(TO_DIR, 'annotations/ann.json'), 'w') as fp:
        json.dump(ann1, fp, indent=4, separators=(',', ': '))

    print("Step 2 is finished!")


def split_dataset():
    """
    将数据集按9:1比例划分为Train / Test Dataset
    """
    print("Step 5 is working ...")

    # step1: 创建相关目录
    path = [os.path.join(TO_DIR, 'train'), os.path.join(TO_DIR, 'test')]
    for _ in path:
        if not os.path.exists(_):
            os.makedirs(os.path.join(_, 'annotations'))
            os.makedirs(os.path.join(_, 'defect'))

    # step2: 划分数据集
    np.random.seed(666)  # set random seed

    total_defect_imgs = os.listdir('./tcdata/imgs_defect')  # 4371
    split_ratio = 0.1  # train : test <---> 9 : 1
    test_nums = int(split_ratio * len(total_defect_imgs))  # 测试集数量：437

    test_idx = np.random.choice(total_defect_imgs, test_nums, replace=False)  # 测试集数量：437
    train_idx = [_ for _ in total_defect_imgs if _ not in test_idx]  # 训练集数量：3934
    # note: 经过fabric2coco转换会过滤掉一些不符条件的图片，因此数量会少于当前 ( < 3934 , < 437 )

    # step3: 拷贝图片至目标路径
    print("\n拷贝训练集图片...")
    for img in tqdm(train_idx):
        shutil.copy(os.path.join('./tcdata/imgs_defect/', img), os.path.join(TO_DIR, 'train/defect'))
    print("\n拷贝测试集图片...")
    for img in tqdm(test_idx):
        shutil.copy(os.path.join('./tcdata/imgs_defect/', img), os.path.join(TO_DIR, 'test/defect'))

    # step4: 生成train|test annotations
    print("生成训练 & 测试标注信息文件...")
    ann_total = mmcv.load('./tcdata/annotations/ann.json')
    ann_train = [_ for _ in ann_total if _['name'] in train_idx]
    ann_test = [_ for _ in ann_total if _['name'] in test_idx]
    json.dump(ann_train, open('./tcdata/train/annotations/ann_train.json', 'w'), indent=4, separators=(',', ': '))
    json.dump(ann_test, open('./tcdata/test/annotations/ann_test.json', 'w'), indent=4, separators=(',', ': '))

    print("Step 5 is finished!")


def main():
    # step 1: copy annotations.
    # copy_annotations()

    # step 2: merge annotations file. (part1 + part2)
    # merge_annotations()

    # step 3: move defect images.
    # move_defect_imgs()

    # step 4: move normal images.
    # move_normal_imgs()

    # step 5: split train/test dataset. (9 : 1)
    # split_dataset()

    # --------------------------------------- #
    print('\nAll work for {} dataset is done.'.format('fabric'))


if __name__ == '__main__':
    main()
