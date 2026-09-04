import copy

def _select(registry, dataset_names):
    return {dataset_name: copy.deepcopy(registry[dataset_name]) for dataset_name in dataset_names}


# libero
LIBERO_NO_NOOPS_DATASETS = {
    "libero_object_no_noops": {
        "data_root": "playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot",
        "data_weight": 1.0,
        "data_class": "lerobot_vla",
        "data_type": "libero",
        "video_keys": ["observation.images.image", "observation.images.wrist_image"],
    },
    "libero_goal_no_noops": {
        "data_root": "playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot",
        "data_weight": 1.0,
        "data_class": "lerobot_vla",
        "data_type": "libero",
        "video_keys": ["observation.images.image", "observation.images.wrist_image"],
    },
    "libero_spatial_no_noops": {
        "data_root": "playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot",
        "data_weight": 1.0,
        "data_class": "lerobot_vla",
        "data_type": "libero",
        "video_keys": ["observation.images.image", "observation.images.wrist_image"],
    },
    "libero_10_no_noops": {
        "data_root": "playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot",
        "data_weight": 1.0,
        "data_class": "lerobot_vla",
        "data_type": "libero",
        "video_keys": ["observation.images.image", "observation.images.wrist_image"],
    },
}

LIBERO_NO_NOOPS_DATASETS_MULTI = {
    "libero_ipec": {
        "data_root": "playground/Datasets/LEROBOT_LIBERO_DATA",
        "data_weight": 1.0,
        "data_class": "multi_lerobot_vla",
        "data_type": "libero",
        "video_keys": ["observation.images.image", "observation.images.wrist_image"],
    }
}

# aloha
ALOHA_NOTAC_3_VIEW_KEYS = [
    "observation.image.third_view",
    "observation.image.left_wrist_view",
    "observation.image.right_wrist_view",
]

ALOHA_TAC_7_VIEW_KEYS = [
    "observation.image.third_view",
    "observation.image.left_wrist_view",
    "observation.image.right_wrist_view",
    "observation.image.left_wrist_left_tactile",
    "observation.image.left_wrist_right_tactile",
    "observation.image.right_wrist_left_tactile",
    "observation.image.right_wrist_right_tactile",
]

ALOHA_WIPE_BOARD = {
    "wipe_the_white_plastic_board_Aloha": {
        "data_root": "playground/Datasets/neoteai/lerobot/wipe_the_white_plastic_board_Aloha",
        "data_weight": 1.0,
        "data_class": "lerobot_vla",
        "data_type": "aloha_tac",
        "video_keys": ALOHA_TAC_7_VIEW_KEYS,
    },
}

ALOHA_WIPE_BOARD_NOTAC = {
    "wipe_the_white_plastic_board_Aloha": {
        "data_root": "playground/Datasets/neoteai/lerobot/wipe_the_white_plastic_board_Aloha",
        "data_weight": 1.0,
        "data_class": "lerobot_vla",
        "data_type": "aloha_tac",
        "video_keys": ALOHA_NOTAC_3_VIEW_KEYS,
    },
}

# flexiv
FLEXIV_NOTAC_2_VIEW_KEYS = [
    "observation.images.third_view",
    "observation.images.left_wrist_view",
]

FLEXIV_TAC_4_VIEW_KEYS = [
    "observation.images.third_view",
    "observation.images.left_wrist_view",
    "observation.images.left_wrist_left_tactile",
    "observation.images.left_wrist_right_tactile",
]

FLEXIV_PLUG_2VIEWS = {
    "plug_outlet_flexiv_0621": {
        "data_root": "playground/Datasets/neoteai/lerobot/plug_outlet_flexiv_0621",
        "data_weight": 1.0,
        "data_class": "lerobot_vla",
        "data_type": "flexiv_tac",
        "video_keys": FLEXIV_NOTAC_2_VIEW_KEYS,
    },
}

FLEXIV_PLUG_4VIEWS = {
    "plug_outlet_flexiv_0621": {
        "data_root": "playground/Datasets/neoteai/lerobot/plug_outlet_flexiv_0621",
        "data_weight": 1.0,
        "data_class": "lerobot_vla",
        "data_type": "flexiv_tac",
        "video_keys": FLEXIV_TAC_4_VIEW_KEYS,
    },
}

FLEXIV_FRUITS_2VIEWS = {
    "fruits_collection_flexiv_0618": {
        "data_root": "playground/Datasets/neoteai/lerobot/fruits_collection_flexiv_0618",
        "data_weight": 1.0,
        "data_class": "multi_lerobot_vla",
        "data_type": "flexiv_tac",
        "video_keys": FLEXIV_NOTAC_2_VIEW_KEYS,
    },
    "fruits_collection_flexiv_0716": {
        "data_root": "playground/Datasets/neoteai/lerobot/fruits_collection_flexiv_0716",
        "data_weight": 1.0,
        "data_class": "multi_lerobot_vla",
        "data_type": "flexiv_tac",
        "video_keys": FLEXIV_NOTAC_2_VIEW_KEYS,
    },
}

FLEXIV_FRUITS_4VIEWS = {
    "fruits_collection_flexiv_0618": {
        "data_root": "playground/Datasets/neoteai/lerobot/fruits_collection_flexiv_0618",
        "data_weight": 1.0,
        "data_class": "multi_lerobot_vla",
        "data_type": "flexiv_tac",
        "video_keys": FLEXIV_TAC_4_VIEW_KEYS,
    },
    "fruits_collection_flexiv_0716": {
        "data_root": "playground/Datasets/neoteai/lerobot/fruits_collection_flexiv_0716",
        "data_weight": 1.0,
        "data_class": "multi_lerobot_vla",
        "data_type": "flexiv_tac",
        "video_keys": FLEXIV_TAC_4_VIEW_KEYS,
    },
}

FLEXIV_LIFT_BOTTLE_2VIEWS = {
    "lift_bottle_flexiv_0626": {
        "data_root": "playground/Datasets/neoteai/lerobot/lift_bottle_flexiv_0626",
        "data_weight": 1.0,
        "data_class": "multi_lerobot_vla",
        "data_type": "flexiv_tac",
        "video_keys": FLEXIV_NOTAC_2_VIEW_KEYS,
    },
    "lift_bottle_flexiv_0718": {
        "data_root": "playground/Datasets/neoteai/lerobot/lift_bottle_flexiv_0718",
        "data_weight": 1.0,
        "data_class": "multi_lerobot_vla",
        "data_type": "flexiv_tac",
        "video_keys": FLEXIV_NOTAC_2_VIEW_KEYS,
    },
}

FLEXIV_LIFT_BOTTLE_4VIEWS = {
    "lift_bottle_flexiv_0626": {
        "data_root": "playground/Datasets/neoteai/lerobot/lift_bottle_flexiv_0626",
        "data_weight": 1.0,
        "data_class": "multi_lerobot_vla",
        "data_type": "flexiv_tac",
        "video_keys": FLEXIV_TAC_4_VIEW_KEYS,
    },
    "lift_bottle_flexiv_0718": {
        "data_root": "playground/Datasets/neoteai/lerobot/lift_bottle_flexiv_0718",
        "data_weight": 1.0,
        "data_class": "multi_lerobot_vla",
        "data_type": "flexiv_tac",
        "video_keys": FLEXIV_TAC_4_VIEW_KEYS,
    },
}

FLEXIV_CLIP_CHIPS_2VIEWS = {
    "clip_chips_flexiv_0708": {
        "data_root": "playground/Datasets/neoteai/lerobot/clip_chips_flexiv_0708",
        "data_weight": 1.0,
        "data_class": "lerobot_vla",
        "data_type": "flexiv_tac",
        "video_keys": FLEXIV_NOTAC_2_VIEW_KEYS,
    }
}

FLEXIV_CLIP_CHIPS_4VIEWS = {
    "clip_chips_flexiv_0708": {
        "data_root": "playground/Datasets/neoteai/lerobot/clip_chips_flexiv_0708",
        "data_weight": 1.0,
        "data_class": "lerobot_vla",
        "data_type": "flexiv_tac",
        "video_keys": FLEXIV_TAC_4_VIEW_KEYS,
    }
}

FLEXIV_INSERT_BOARD_2VIEWS = {
    "insert_board_flexiv_0720": {
        "data_root": "playground/Datasets/neoteai/lerobot/insert_board_flexiv_0720",
        "data_weight": 1.0,
        "data_class": "lerobot_vla",
        "data_type": "flexiv_tac",
        "video_keys": FLEXIV_NOTAC_2_VIEW_KEYS,
    }
}

FLEXIV_INSERT_BOARD_4VIEWS = {
    "insert_board_flexiv_0720": {
        "data_root": "playground/Datasets/neoteai/lerobot/insert_board_flexiv_0720",
        "data_weight": 1.0,
        "data_class": "lerobot_vla",
        "data_type": "flexiv_tac",
        "video_keys": FLEXIV_TAC_4_VIEW_KEYS,
    }
}

FLEXIV_INSERT_BOARD_NEW_2VIEWS = {
    "insert_board_flexiv_0720": {
        "data_root": "playground/Datasets/neoteai/lerobot/insert_board_flexiv_0720-new",
        "data_weight": 1.0,
        "data_class": "lerobot_vla",
        "data_type": "flexiv_tac",
        "video_keys": FLEXIV_NOTAC_2_VIEW_KEYS,
    }
}

FLEXIV_INSERT_BOARD_NEW_4VIEWS = {
    "insert_board_flexiv_0720": {
        "data_root": "playground/Datasets/neoteai/lerobot/insert_board_flexiv_0720-new",
        "data_weight": 1.0,
        "data_class": "lerobot_vla",
        "data_type": "flexiv_tac",
        "video_keys": FLEXIV_TAC_4_VIEW_KEYS,
    }
}

DATASET_NAMED_MIXTURES = {
    # libero
    "libero_all": _select(LIBERO_NO_NOOPS_DATASETS, tuple(LIBERO_NO_NOOPS_DATASETS)),
    "libero_all_multi": _select(LIBERO_NO_NOOPS_DATASETS_MULTI, tuple(LIBERO_NO_NOOPS_DATASETS_MULTI)),

    # aloha
    "aloha_wipe_board": _select(ALOHA_WIPE_BOARD, tuple(ALOHA_WIPE_BOARD)),
    "aloha_wipe_board_notac": _select(ALOHA_WIPE_BOARD_NOTAC, tuple(ALOHA_WIPE_BOARD_NOTAC)),

    # flexiv
    "flexiv_plug_2views": _select(FLEXIV_PLUG_2VIEWS, tuple(FLEXIV_PLUG_2VIEWS)),
    "flexiv_plug_4views": _select(FLEXIV_PLUG_4VIEWS, tuple(FLEXIV_PLUG_4VIEWS)),
    "flexiv_fruits_2views": _select(FLEXIV_FRUITS_2VIEWS, tuple(FLEXIV_FRUITS_2VIEWS)),
    "flexiv_fruits_4views": _select(FLEXIV_FRUITS_4VIEWS, tuple(FLEXIV_FRUITS_4VIEWS)),
    "flexiv_lift_bottle_2views": _select(FLEXIV_LIFT_BOTTLE_2VIEWS, tuple(FLEXIV_LIFT_BOTTLE_2VIEWS)),
    "flexiv_lift_bottle_4views": _select(FLEXIV_LIFT_BOTTLE_4VIEWS, tuple(FLEXIV_LIFT_BOTTLE_4VIEWS)),
    "flexiv_clip_chips_2views": _select(FLEXIV_CLIP_CHIPS_2VIEWS, tuple(FLEXIV_CLIP_CHIPS_2VIEWS)),
    "flexiv_clip_chips_4views": _select(FLEXIV_CLIP_CHIPS_4VIEWS, tuple(FLEXIV_CLIP_CHIPS_4VIEWS)),
    "flexiv_insert_board_2views": _select(FLEXIV_INSERT_BOARD_2VIEWS, tuple(FLEXIV_INSERT_BOARD_2VIEWS)),
    "flexiv_insert_board_4views": _select(FLEXIV_INSERT_BOARD_4VIEWS, tuple(FLEXIV_INSERT_BOARD_4VIEWS)),
    "flexiv_insert_board_2views_new": _select(FLEXIV_INSERT_BOARD_NEW_2VIEWS, tuple(FLEXIV_INSERT_BOARD_NEW_2VIEWS)),
    "flexiv_insert_board_4views_new": _select(FLEXIV_INSERT_BOARD_NEW_4VIEWS, tuple(FLEXIV_INSERT_BOARD_NEW_4VIEWS)),
}


def num_video_keys(data_mix):
    """Number of camera/video keys declared by a named data mix (raises if unknown)."""
    mixture = DATASET_NAMED_MIXTURES[str(data_mix)]
    return max(len(spec["video_keys"]) for spec in mixture.values())


def video_keys(data_mix):
    """Camera/video keys of a named data mix (longest spec, consistent with num_video_keys)."""
    mixture = DATASET_NAMED_MIXTURES[str(data_mix)]
    return max((spec["video_keys"] for spec in mixture.values()), key=len)
