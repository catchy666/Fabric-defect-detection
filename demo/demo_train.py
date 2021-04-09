import json
import os
import cv2
import time
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from pycocotools.coco import COCO
from argparse import ArgumentParser

"""
defect_name2label = {
    '沾污': 1, '错花': 2, '水印': 3, '花毛': 4, '缝头': 5, '缝头印': 6, '虫粘': 7, '破洞': 8, '褶子': 9,
    '织疵': 10, '漏印': 11, '蜡斑': 12, '色差': 13, '网折': 14, '其他': 15
}
"""

CONFIG = '../configs/fabric/cascade_rcnn_r50_fpn_50e_coco.py'
CHECKPOINTS = ''
ANN_FILE = '../tcdata/train/annotations/instance_train.json'


def demo_predict(img_name):
    model = init_detector(CONFIG, CHECKPOINTS)
    result = inference_detector(model, img_name)
    # print(result)
    show_result_pyplot(model, img_name, result)


def draw_ground_truth(img_name):
    coco = COCO(ANN_FILE)
    ann_ids = coco.getAnnIds(imgIds=get_img_id(ANN_FILE, os.path.basename(img_name)), iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    image = cv2.imread(img_name)
    for i in range(len(anns)):
        x, y, w, h = anns[i]['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255))
        cv2.putText(image,
                    str(anns[i]['category_id']),
                    (x, y - 2),
                    cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=2, color=(0, 0, 255))
    cv2.imwrite('_truth.jpg', image)


def get_img_id(ann, name):
    with open(ann, 'r') as f:
        res = json.load(f)
    for img in res["images"]:
        if name == img["file_name"]:
            return img["id"]


def main():
    parser = ArgumentParser(description='Predict & Draw GTruth')
    parser.add_argument('--img', help='image file path',
                        default='../tcdata/train/defect/0917B2_b4dd1dcddfcef8722201909171928043.jpg'
                        )
    args = parser.parse_args()

    # option 1:
    demo_predict(img_name=args.img)

    # option 2:
    draw_ground_truth(img_name=args.img)


if __name__ == '__main__':
    main()
