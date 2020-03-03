import json_tricks as json
import numpy as np

# create train/val split
# 分别处理训练子集/验证子集/测试子集，格式见mpii_test.json
file_name = 'mpii_valid.json'
with open(file_name) as anno_file:
    anno = json.load(anno_file)

gt_db = []
for a in anno:
    # 详见mpii_README.md
    image_name = a['image']  # 图像名称 image

    c = np.array(a['center'], dtype=np.float)  # center: 图中人体的大致中心坐标
    s = np.array([a['scale'], a['scale']], dtype=np.float)  # scale: 人体框的高度/200像素

    # Adjust center/scale slightly to avoid cropping limbs
    # 太高人体中心位置，放大人体尺度，避免裁剪到边缘
    if c[0] != -1:
        c[1] = c[1] + 15 * s[1]
        s = s * 1.25

    # MPII uses matlab format, index is based 1,
    # we should first convert to 0-based index
    # matlab格式的角标是以1开头的
    c = c - 1

    joints_3d = np.zeros((16, 3), dtype=np.float)
    joints_3d_vis = np.zeros((16, 3), dtype=np.float)

    joints = np.array(a['joints'])
    joints[:, 0:2] = joints[:, 0:2] - 1
    joints_vis = np.array(a['joints_vis'])

    joints_3d[:, 0:2] = joints[:, 0:2]
    joints_3d_vis[:, 0] = joints_vis[:]
    joints_3d_vis[:, 1] = joints_vis[:]