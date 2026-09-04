from starVLA.dataloader.vla.data_config.data_config_base import BaseDataConfig, BaseDeltaDataConfig, BaseRelativeDataConfig
from starVLA.dataloader.vla.data_config.data_config import AlohaTacDataConfig, LiberoDataConfig, FlexivTacDataConfig

ROBOT_TYPE_CONFIG_MAP = {
    "libero": LiberoDataConfig(),
    "aloha_tac": AlohaTacDataConfig(),
    "flexiv_tac": FlexivTacDataConfig(),
}
