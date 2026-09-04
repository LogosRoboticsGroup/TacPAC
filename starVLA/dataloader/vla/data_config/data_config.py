# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from PIL import Image

from starVLA.dataloader.vla.data_config.data_config_base import BaseDataConfig, BaseDeltaDataConfig, BaseRelativeDataConfig

class LiberoDataConfig(BaseDataConfig):
    video_keys = ['observation.images.image', 'observation.images.wrist_image']
    state_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    action_ids = [0, 1, 2, 3, 4, 5, 6]
    action_origin_dim = 7
    gripper_state_ids = [6, 7]
    gripper_action_ids = [6]
    disable_state = True


class AlohaTacDataConfig(BaseRelativeDataConfig):
    statistics_cache_key = "aloha_tac_relative"
    video_keys = [
        "observation.image.third_view",
        "observation.image.left_wrist_view",
        "observation.image.right_wrist_view",
        "observation.image.left_wrist_left_tactile",
        "observation.image.left_wrist_right_tactile",
        "observation.image.right_wrist_left_tactile",
        "observation.image.right_wrist_right_tactile",
    ]
    state_ids = list(range(14))
    action_ids = list(range(14))
    action_origin_dim = 14
    gripper_state_ids = [6, 13]
    gripper_action_ids = [6, 13]
    transformed_action_dim = 14
    use_all_columns = True
    disable_state = False

class FlexivTacDataConfig(BaseRelativeDataConfig):
    statistics_cache_key = "flexiv_tac_relative"
    video_keys = [
        "observation.images.third_view",
        "observation.images.second_third_view",
        "observation.images.left_wrist_view",
        "observation.images.left_wrist_left_tactile",
        "observation.images.left_wrist_right_tactile",
    ]
    state_ids = list(range(8))
    action_ids = list(range(8))
    action_origin_dim = 8
    gripper_state_ids = [7,]
    gripper_action_ids = [7,]
    transformed_action_dim = 8
    use_all_columns = True
    disable_state = False

