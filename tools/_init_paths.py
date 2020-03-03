# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


# 将路径path添加进python搜索路径中
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

# 所在脚本是以完整路径被运行的， 那么将输出该脚本所在的完整路径
# 所在脚本是以相对路径被运行的， 那么将输出空目录
this_dir = osp.dirname(__file__)

# 将lib文件夹加入搜索路径
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

mm_path = osp.join(this_dir, '..', 'lib/poseeval/py-motmetrics')
add_path(mm_path)
