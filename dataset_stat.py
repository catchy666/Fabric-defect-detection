import matplotlib.pyplot as plt
import mmcv


anns = mmcv.load('./tcdata/annotations/ann.json')

classes = []
for ann in anns:
    classes.append(ann['defect_name'])

classes = list(set(classes))

# Tile classes
CATEGORY = {}
for _ in classes:
    CATEGORY[_] = 0

# calculate the number of label
for ann in anns:
    CATEGORY[ann['defect_name']] += 1

y = [v for v in CATEGORY.values()]  # [576, 2151, 2174, 1112, 8642, 331]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.bar(range(1, len(CATEGORY.keys()) + 1), y, color='rgb', tick_label=classes)
for a, b in zip(range(1, len(CATEGORY.keys()) + 1), y):
    plt.text(a, b + 0.05, b, ha='center', va='bottom', fontsize=10)

plt.savefig('./tcdata/annotations/ann_stat.png')  # save stat fig
plt.show()
