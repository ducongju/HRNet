# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json_tricks as json
from collections import OrderedDict

import numpy as np
from scipy.io import loadmat, savemat

from dataset.JointsDataset import JointsDataset

logger = logging.getLogger(__name__)


class MPIIDataset(JointsDataset):
    """
    { 0 - r ankle,
    1 - r knee,
    2 - r hip,
    3 - l hip,
    4 - l knee,
    5 - l ankle,
    6 - pelvis,
    7 - thorax,
    8 - upper neck,
    9 - head top,
    10 - r wrist,
    11 - r elbow,
    12 - r shoulder,
    13 - l shoulder,
    14 - l elbow,
    15 - l wrist }
    """

    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        # 示意图见mpii.vsdx
        self.num_joints = 16
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    # 读取json文件并返回处理后的groundtruth_database
    def _get_db(self):
        # create train/val split
        # 分别处理训练子集/验证子集/测试子集，格式见mpii_test.json
        file_name = os.path.join(
            self.root, 'annot', self.image_set + '.json'
        )
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            # 详见mpii_README.md
            image_name = a['image']  # 图像名称 image = {str}

            c = np.array(a['center'], dtype=np.float)  # 图中人体的大致中心坐标 center = {ndarray: (2,)}
            # 把框设置为正方形
            # TODO 处理细节：是否可以把框设置为其他形状
            s = np.array([a['scale'], a['scale']], dtype=np.float)  # 人体框的高度/200像素 scale = {float64}

            # Adjust center/scale slightly to avoid cropping limbs
            # 太高人体中心位置，放大人体尺度，避免裁剪到边缘
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            # matlab格式的角标是以1开头的
            c = c - 1

            # TODO 为了推广适用于到三维而增加了一维0
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            if self.image_set != 'test':
                joints = np.array(a['joints'])  # 所有关节的坐标 joints = {ndarray: (16, 2)}
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(a['joints_vis'])  # 关节是否可见 joints_vis = {ndarray: (16,)}
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            gt_db.append(
                {
                    'image': os.path.join(self.root, image_dir, image_name),
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                }
            )

        return gt_db

    # 利用testset的预测关节坐标结果，评估PCKh指标
    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join(cfg.DATASET.ROOT,
                               'annot',
                               'gt_{}.mat'.format(cfg.DATASET.TEST_SET))
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']  # 关节名称 dataset_joints = {ndarray: (1, 16)}
        jnt_missing = gt_dict['jnt_missing']  # 是否未标注关节 jnt_missing = {ndarray: (16, 2958)}
        pos_gt_src = gt_dict['pos_gt_src']  # 实际关节坐标 pos_gt_src = {ndarray: (16, 2, 2958)}
        headboxes_src = gt_dict['headboxes_src']  # 头部边界框左上角和右下角坐标 headboxes_src = {ndarray: (2, 2, 2958)}

        # batch*num_joints*coordinate变为对应于.mat格式
        pos_pred_src = np.transpose(preds, [1, 2, 0])  # 预测关节位置 pos_pred_src = {ndarray: (16, 2, 2958)}

        # 提取角标
        # head_all = np.where(dataset_joints == 'head')  # head_all: {tuple: 2}
        head = np.where(dataset_joints == 'head')[1][0]  # 头部序号 head = {int64}:9
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing  # 是否标注了关节 jnt_visible = {ndarray: (16, 2958)}
        uv_error = pos_pred_src - pos_gt_src  # 关节位置误差 uv_error = {ndarray: (16, 2, 2958)}
        # linalg=linear（线性）+algebra（代数），norm则表示范数，默认2范数
        uv_err = np.linalg.norm(uv_error, axis=1)  # 误差距离(对坐标求2范数) uv_err = {ndarray: (16, 2958)}
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]  # 头部尺寸 headsizes = {ndarray: (2, 2958)}
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS  # TODO 处理细节：以头部框对角线距离作为归一化参考 headsizes = {ndarray: (2958,)}
        # np.multiply: 数组和矩阵对应位置相乘，输出与相乘数组/矩阵的大小一致
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))  # 每个关节都参照头 scale = {ndarray: (16, 2958)}
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)  # 归一化结果 scaled_uv_err = {ndarray: (16, 2958)}
        jnt_count = np.sum(jnt_visible, axis=1)  # 每个关节可见的总数 jnt_count = {ndarray: (16,)}
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)  # 关节是否低于门限 less_than_threshold = {ndarray: (16, 2958)}
        # 每个关节归一化超过0.5的百分比概率 PCKh = {ndarray: (16,)}
        PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5 + 0.01, 0.01)  # 门限设置从0到0.5
        pckAll = np.zeros((len(rng), 16))  # 每个关节归一化超过各门限的百分比概率 PCKALL = {ndarray: (50, 16)}

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100. * np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True  # TODO 处理细节：把骨盆、胸部、上颈给mask掉了。。

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),  # TODO 处理细节：对称关节一起计算
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)  # 创建有序字典

        return name_value, name_value['Mean']
