from vision_toolkit2.config import Config
from vision_toolkit2.serie import Serie

from dataclasses import dataclass


@dataclass
class BaseSegmentationResults:
    input: Serie
    config: Config
