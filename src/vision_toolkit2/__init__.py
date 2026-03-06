from .segmentation.base_segmentation import Segmentation
from .config import Config, StackedConfig
from .serie import Serie

def example_dataset_dir():
    return f'{__file__.replace("__init__.py", "")}/example_dataset'

__all__ = [
    Segmentation,
    Config,
    StackedConfig,
    Serie,
]
