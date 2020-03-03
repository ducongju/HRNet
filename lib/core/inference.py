# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from utils.transforms import transform_preds


# 输入: 关节热图 batch_heatmaps = {ndarray: (batch_size, num_joints, height, width)}
# 输出: 预测位置 preds = {ndarray: (batch_size, num_joints, 2)}
# 预测值 maxvals = {ndarray: (batch_size, num_joints, 1)}
def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    # heatmaps_reshaped = {ndarray: (batch_size, num_joints, height*width)}
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))

    # 找到每个batch每个关节的最大值和对应索引
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    # 将预测索引转化成生成预测位置坐标x,y
    # preds = {ndarray: (batch_size, num_joints, 2)}
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    # 生成预测值大于0的mask
    # pred_mask = {ndarray: (batch_size, num_joints, 2)}
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    # 只有大于0的预测值的预测位置才是有效位置
    preds *= pred_mask
    return preds, maxvals


# 输入: 配置文件 config = lib/config/defalut.py
# 关节热图 batch_heatmaps = {ndarray: (batch_size, num_joints, height, width)}
# 图中人体的大致中心坐标 center = {ndarray: (batch_size, 2)}
# 人体框的高度/200像素 scale = {ndarray: (batch_size)}
# TODO 输出:
def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                # 单张热图 hm = {ndarray: (height, width)}
                hm = batch_heatmaps[n][p]
                # TODO: 本来就是整数为什么要四舍五入
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))  # 热图某关节最大值的坐标 px, py
                # 预测的位置需要在heatmap内部
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    # 计算最大值位置上方像素与下方像素的差值，同理右方与左方
                    # 差值 diff = {ndarray: (2,)}
                    diff = np.array(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ]
                    )  # py, px是因为对应的是width, height
                    # 取最大值对应坐标到次大值对应坐标四分之一处为最终坐标
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    # TODO: 这里返回的最终预测格式是什么？
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals
