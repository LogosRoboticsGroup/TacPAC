# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Shared helpers for loading StarVLA checkpoints in inference servers."""

from __future__ import annotations

import logging
from typing import Any

from starVLA.model.framework.base_framework import baseframework


INFERENCE_CONFIG_OVERRIDES: dict[str, Any] = {
    "framework.skip_dit_load_from_pretrain": True,
    "framework.action_model.skip_load_from_pretrain": True,
}


def load_framework_for_inference(ckpt_path: str):
    """Build a framework for inference and let the checkpoint provide DiT weights."""
    logging.info(
        "Loading framework for inference with pretrained DiT loads disabled; "
        "checkpoint weights will be applied from %s",
        ckpt_path,
    )
    return baseframework.from_pretrained(
        ckpt_path,
        config_overrides=INFERENCE_CONFIG_OVERRIDES,
    )
