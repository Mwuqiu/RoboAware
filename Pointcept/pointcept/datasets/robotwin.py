from .builder import DATASETS
from .so100_arm import So100Dataset


@DATASETS.register_module()
class RobotWinDataset(So100Dataset):
    """Frame-level npy robot segmentation dataset exported from RoboTwin."""

    pass
