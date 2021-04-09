# for cascade rcnn
import torch
from icecream import ic

model_name = "../pretrained/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"
model = torch.load(model_name)

# weight
num_classes = 15 + 1

# for faster model
model["state_dict"]["roi_head.bbox_head.fc_cls.weight"] = model["state_dict"]["roi_head.bbox_head.fc_cls.weight"][
                                                          :num_classes, :]
model["state_dict"]["roi_head.bbox_head.fc_cls.bias"] = model["state_dict"]["roi_head.bbox_head.fc_cls.bias"][
                                                        :num_classes]

# for cascade model
# model["state_dict"]["roi_head.bbox_head.0.fc_cls.weight"] = model["state_dict"]["roi_head.bbox_head.0.fc_cls.weight"][
#                                                             :num_classes, :]
# model["state_dict"]["roi_head.bbox_head.1.fc_cls.weight"] = model["state_dict"]["roi_head.bbox_head.1.fc_cls.weight"][
#                                                             :num_classes, :]
# model["state_dict"]["roi_head.bbox_head.2.fc_cls.weight"] = model["state_dict"]["roi_head.bbox_head.2.fc_cls.weight"][
#                                                             :num_classes, :]
# # bias
# model["state_dict"]["roi_head.bbox_head.0.fc_cls.bias"] = model["state_dict"]["roi_head.bbox_head.0.fc_cls.bias"][
#                                                           :num_classes]
# model["state_dict"]["roi_head.bbox_head.1.fc_cls.bias"] = model["state_dict"]["roi_head.bbox_head.1.fc_cls.bias"][
#                                                           :num_classes]
# model["state_dict"]["roi_head.bbox_head.2.fc_cls.bias"] = model["state_dict"]["roi_head.bbox_head.2.fc_cls.bias"][
#                                                           :num_classes]

# save new model
torch.save(model, "../pretrained/modified_c16_faster_rcnn_r50_fpn_2x_coco.pth")
