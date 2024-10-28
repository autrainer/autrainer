from .abstract_dataset import (
    AbstractDataset,
    BaseClassificationDataset,
    BaseMLClassificationDataset,
    BaseMTRegressionDataset,
    BaseRegressionDataset,
)
from .aibo import AIBO
from .dcase_2016_t1 import DCASE2016Task1
from .dcase_2018_t3 import DCASE2018Task3
from .dcase_2020_t1a import DCASE2020Task1A
from .edansa2019 import EDANSA2019
from .emo_db import EmoDB
from .speech_commands import SpeechCommands
from .toy_dataset import ToyDataset


__all__ = [
    "AbstractDataset",
    "BaseClassificationDataset",
    "BaseMLClassificationDataset",
    "BaseMTRegressionDataset",
    "BaseRegressionDataset",
    "AIBO",
    "DCASE2016Task1",
    "DCASE2018Task3",
    "DCASE2020Task1A",
    "EDANSA2019",
    "EmoDB",
    "SpeechCommands",
    "ToyDataset",
]
