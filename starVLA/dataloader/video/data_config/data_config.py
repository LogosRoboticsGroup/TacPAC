from starVLA.dataloader.video.data_config.data_config_base import VideoBaseDataConfig


class LIBERO_DataConfig(VideoBaseDataConfig):
    video_keys = ("observation.images.image", "observation.images.wrist_image")

class ALOHA_Tac_DataConfig(VideoBaseDataConfig):
    video_keys = ("observation.image.right_wrist_view", "observation.image.right_wrist_left_tactile", "observation.image.right_wrist_right_tactile")