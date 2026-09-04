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


class BaseDataConfig(ABC):
    video_keys = ['observation.images.image', 'observation.images.wrist_image']
    state_ids = [0, 1, 2, 3, 4, 5, 6]
    action_ids = [0, 1, 2, 3, 4, 5, 6]
    action_origin_dim = 7
    gripper_state_ids = [6]
    gripper_action_ids = [6]
    state_pad_size = 7
    action_pad_size = 7
    transformed_action_dim = None
    disable_state = False
    binarize_gripper = True
    use_all_columns = False  # When True, skip state_ids/action_ids indexing and gripper binarization
    norm_eps = 1e-6

    # Image processing settings
    image_size = (224, 224)  # Default image size (H, W)
    _pixel_transforms_resize = None  # Lazy initialized

    def set_image_size(self, image_size):
        """Set image size and reset transform cache."""
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = tuple(image_size)
        self._pixel_transforms_resize = None  # Reset cache

    @property
    def pixel_transforms_resize(self):
        """Lazy initialize pixel transforms."""
        if self._pixel_transforms_resize is None:
            self._pixel_transforms_resize = transforms.Compose([
                transforms.Resize(min(self.image_size)), 
                transforms.CenterCrop(self.image_size)
            ])
        return self._pixel_transforms_resize
    
    def resize_image(self, img):
        """
        Resize a single image or batch of images to target size.
        
        Args:
            img: PIL.Image, torch.Tensor [C, H, W] or [H, W, C] or [B, C, H, W], or np.ndarray
        Returns:
            torch.Tensor [C, H, W] or [B, C, H, W] (preserves batch dimension if present)
        """
        if isinstance(img, Image.Image):
            img = TF.pil_to_tensor(img)
        elif isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[-1] in [1, 3, 4]:  # HWC format
                img = torch.from_numpy(img).permute(2, 0, 1)
            elif img.ndim == 4 and img.shape[1] in [1, 3, 4]:  # BCHW format
                img = torch.from_numpy(img)
            elif img.ndim == 4 and img.shape[-1] in [1, 3, 4]:  # BHWC format
                img = torch.from_numpy(img).permute(0, 3, 1, 2)
            else:
                img = torch.from_numpy(img)
        elif isinstance(img, torch.Tensor):
            if img.ndim == 3 and img.shape[-1] in [1, 3, 4] and img.shape[0] not in [1, 3, 4]:
                # Likely HWC format, convert to CHW
                img = img.permute(2, 0, 1)
            elif img.ndim == 4 and img.shape[-1] in [1, 3, 4] and img.shape[1] not in [1, 3, 4]:
                # Likely BHWC format, convert to BCHW
                img = img.permute(0, 3, 1, 2)
        
        # torchvision transforms support both [C, H, W] and [B, C, H, W]
        return self.pixel_transforms_resize(img)

    def resize_images(self, images):
        """
        Resize a list of images.
        
        Args:
            images: List of images (PIL.Image, torch.Tensor, or np.ndarray)
        Returns:
            List of torch.Tensor [C, H, W]
        """
        return [self.resize_image(img) for img in images]
    
    def select_state_columns(self, state):
        state_array = np.asarray(state)
        if self.use_all_columns:
            return state_array
        if state_array.shape[-1] == len(self.state_ids):
            return state_array
        return state_array[..., self.state_ids]

    def _minmax_normalize(self, values, low, high):
        scale = high - low
        safe_scale = np.where(np.abs(scale) > self.norm_eps, scale, 1.0)
        normalized = 2.0 * (values - low) / safe_scale - 1.0
        return np.where(np.abs(scale) > self.norm_eps, normalized, 0.0)

    def _minmax_unnormalize(self, values, low, high):
        values = np.clip(values, -1.0, 1.0)
        return 0.5 * (values + 1.0) * (high - low) + low


    def input_transform_dataloader(self, data, step_idx, action_horizon, inplace=True):
        if inplace:
            data_ = data
        else:
            data_ = {}

        # Resize images if present
        if 'image' in data:
            data_['image'] = self.resize_images(data['image'])

        if self.disable_state:
            if 'state' in data_:
                del data_['state']
        else:
            state = data['state']
            if self.use_all_columns:
                state_transformed = state[step_idx:step_idx + 1]
            else:
                state_transformed = state[step_idx:step_idx + 1, self.state_ids]
            data_['state'] = state_transformed

        action = data['actions']
        if self.use_all_columns:
            action_transformed = action[step_idx:step_idx + action_horizon]
        else:
            action_transformed = action[step_idx:step_idx + action_horizon, self.action_ids]
        data_['num_valid_actions'] = action_transformed.shape[0]
        if action_transformed.shape[0] < action_horizon:
            action_transformed = np.concatenate([action_transformed, action_transformed[-1:].repeat(action_horizon - action_transformed.shape[0], axis=0)], axis=0)
        data_['actions'] = action_transformed
        return data_

    def input_transform(self, data):
        """Transform data for inference (batch of samples)."""
        # Resize images if present (for inference)
        if 'batch_images' in data:
            # data['batch_images'] is a list of samples, each sample is a list of images
            data['batch_images'] = [self.resize_images(imgs) for imgs in data['batch_images']]

        if self.disable_state:
            data['state'] = None
        elif not self.use_all_columns:
            data['state'] = [self.select_state_columns(s) for s in data['state']]
        if 'actions' in data and not self.use_all_columns:
            data['actions'] = [a[..., self.action_ids] for a in data['actions']]
        return data
    
    def output_transform(self, data, input_data):
        if self.use_all_columns:
            return data
        actions = np.zeros((*data['actions'].shape[:-1], self.action_origin_dim), dtype=data['actions'].dtype)
        actions[..., self.action_ids] = data['actions']
        data['actions'] = actions
        return data
    
    def normalize_data(self, data, stats):
        if not self.disable_state:
            unnormalized_state = np.array(data['state'])
            state_norm_stats = stats['state']
            state_min = np.array(state_norm_stats['min'])
            state_max = np.array(state_norm_stats['max'])
            normalized_state = self._minmax_normalize(unnormalized_state, state_min, state_max)
            if self.binarize_gripper and not self.use_all_columns and self.gripper_state_ids:
                normalized_state[..., self.gripper_state_ids] = np.where(normalized_state[..., self.gripper_state_ids] < 0.0, -1, 1)
            data['state'] = normalized_state

        if 'actions' in data:
            unnormalized_actions = np.array(data['actions'])
            action_norm_stats = stats['action']
            action_min = np.array(action_norm_stats['min'])
            action_max = np.array(action_norm_stats['max'])
            normalized_actions = self._minmax_normalize(unnormalized_actions, action_min, action_max)
            if self.binarize_gripper and not self.use_all_columns and self.gripper_action_ids:
                normalized_actions[..., self.gripper_action_ids] = np.where(normalized_actions[..., self.gripper_action_ids] < 0.0, -1, 1)
            data['actions'] = normalized_actions
        return data
    
    def unnormalize_data(self, data, stats):
        normalized_actions = np.array(data['normalized_actions'])
        action_norm_stats = stats['action']
        action_min = np.array(action_norm_stats['min'])
        action_max = np.array(action_norm_stats['max'])
        unnormalized_actions = self._minmax_unnormalize(normalized_actions, action_min, action_max)
        if self.binarize_gripper and not self.use_all_columns and self.gripper_action_ids:
            unnormalized_actions[..., self.gripper_action_ids] = np.where(normalized_actions[..., self.gripper_action_ids] < 0.0, action_min[self.gripper_action_ids], action_max[self.gripper_action_ids])
        data['actions'] = unnormalized_actions
        return data
    
    def pad_data(self, data):
        if data.get('state') is not None:
            state = data['state']
            state = torch.as_tensor(state.astype(np.float32))
            state = F.pad(state, (0, self.state_pad_size - state.shape[-1]))
            data['state'] = state
    
        if data.get('actions') is not None:
            actions = data['actions']
            action_mask = torch.zeros(*actions.shape[:-1], self.action_pad_size, dtype=torch.bool)
            action_mask[..., :actions.shape[-1]] = 1
            num_valid_actions = data.pop('num_valid_actions', None)
            if num_valid_actions is not None:
                # Time-padded steps carry synthetic targets; keep them out of the loss.
                action_mask[..., num_valid_actions:, :] = 0
            data['action_mask'] = action_mask
            
            actions = torch.as_tensor(actions.astype(np.float32))
            actions = F.pad(actions, (0, self.action_pad_size - actions.shape[-1]))
            data['actions'] = actions

        return data

    def unpad_data(self, data):
        if 'normalized_actions' in data:
            normalized_actions = data['normalized_actions']
            action_dim = self.transformed_action_dim
            if action_dim is None and not self.use_all_columns:
                action_dim = len(self.action_ids)
            if action_dim is not None:
                data['normalized_actions'] = normalized_actions[..., :action_dim]
        return data
    
class BaseRelativeDataConfig(BaseDataConfig):
    """Actions are relative to the initial state: action_rel[t] = action_abs[t] - state[0]."""
    def input_transform_dataloader(self, data, step_idx, action_horizon, inplace=True):
        if inplace:
            data_ = data
        else:
            data_ = {}

        if 'image' in data:
            data_['image'] = self.resize_images(data['image'])

        state = data['state'][step_idx:step_idx + 1, self.state_ids]
        data_['state'] = state

        action = data['actions'][step_idx:step_idx + action_horizon, self.action_ids]
        action_transformed = action - state  # broadcast (1, D) over (H, D)
        # Gripper values should be absolute, not relative
        if self.gripper_action_ids:
            action_transformed[:, self.gripper_action_ids] = action[:, self.gripper_action_ids]
        data_['num_valid_actions'] = action_transformed.shape[0]
        if action_transformed.shape[0] < action_horizon:
            # Pad by holding the last pose; zero would mean "snap back to the window origin" in relative space.
            action_transformed = np.concatenate([action_transformed, action_transformed[-1:].repeat(action_horizon - action_transformed.shape[0], axis=0)], axis=0)
        data_['actions'] = action_transformed
        return data_

    def input_transform(self, data):
        # Resize images if present (for inference)
        if 'batch_images' in data:
            data['batch_images'] = [self.resize_images(imgs) for imgs in data['batch_images']]

        data['state'] = np.array([self.select_state_columns(s) for s in data['state']])
        if 'actions' in data:
            actions = np.array([a[:, self.action_ids] for a in data['actions']])
            action_transformed = actions - data['state']  # broadcast (B, 1, D) over (B, H, D)
            if self.gripper_action_ids:
                action_transformed[..., self.gripper_action_ids] = actions[..., self.gripper_action_ids]
            data['actions'] = action_transformed
        return data

    def output_transform(self, data, input_data):
        actions_rel = data['actions']
        state = self.select_state_columns(np.array(input_data['state']))
        actions = state + actions_rel
        # Gripper values are absolute — use raw predictions
        if self.gripper_action_ids:
            actions[..., self.gripper_action_ids] = actions_rel[..., self.gripper_action_ids]
        actions_origin = np.zeros((*actions.shape[:-1], self.action_origin_dim), dtype=actions.dtype)
        actions_origin[..., self.action_ids] = actions
        data['actions'] = actions_origin
        return data


class BaseDeltaDataConfig(BaseDataConfig):
    def input_transform_dataloader(self, data, step_idx, action_horizon, inplace=True):
        if inplace:
            data_ = data
        else:
            data_ = {}
        
        if 'image' in data:
            data_['image'] = self.resize_images(data['image'])
        
        state = data['state'][step_idx:step_idx + 1, self.state_ids]
        state_transformed = state
        data_['state'] = state_transformed
        
        action = data['actions']
        action = np.concatenate([state, action[step_idx:step_idx + action_horizon, self.action_ids]], axis=0)
        action_transformed = np.diff(action, axis=0)
        # Gripper values should be absolute, not delta
        if self.gripper_action_ids:
            action_transformed[:, self.gripper_action_ids] = action[1:, self.gripper_action_ids]
        data_['num_valid_actions'] = action_transformed.shape[0]
        if action_transformed.shape[0] < action_horizon:
            # Zero delta = hold pose, a sensible pad value for delta actions.
            action_transformed = np.concatenate([action_transformed, np.zeros((action_horizon-action_transformed.shape[0], action_transformed.shape[1]))], axis=0)
        data_['actions'] = action_transformed
        return data_

    def input_transform(self, data):
        # Resize images if present (for inference)
        if 'batch_images' in data:
            # data['batch_images'] is a list of samples, each sample is a list of images
            data['batch_images'] = [self.resize_images(imgs) for imgs in data['batch_images']]

        data['state'] = np.array([self.select_state_columns(s) for s in data['state']])
        if 'actions' in data:
            actions_prefix = np.array([actions_prefix[:, self.action_ids] for actions_prefix in data['actions']])
            actions = np.concatenate([data['state'], actions_prefix], axis=-2)
            action_transformed = np.diff(actions, axis=-2)
            action_transformed[..., self.gripper_action_ids] = actions[..., 1:, self.gripper_action_ids]
            data['actions'] = action_transformed
        return data

    def output_transform(self, data, input_data):
        actions_ = data['actions']
        state = self.select_state_columns(np.array(input_data['state']))
        actions = state + np.cumsum(actions_, axis=-2)
        # Gripper values are absolute, not delta — use raw predictions
        if self.gripper_action_ids:
            actions[..., self.gripper_action_ids] = actions_[..., self.gripper_action_ids]
        actions_origin = np.zeros((*actions.shape[:-1], self.action_origin_dim), dtype=actions.dtype)
        actions_origin[..., self.action_ids] = actions
        data['actions'] = actions_origin
        return data
