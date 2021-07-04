from typing import Tuple
from pathlib import Path
from numpy import pad
import torch
from torch import Tensor, nn
from torchaudio.transforms import MFCC
from pandas import DataFrame
from torch.utils.data import Dataset
from src.utils.text import TextUtility
from src.utils.audio import AudioUtility


class CommonVoice(Dataset, TextUtility, AudioUtility):

    def __init__(self, path: Path, df: DataFrame, config) -> None:
        super().__init__(config=config)
        # common voice .csv file
        self.df: DataFrame = df
        self.path = path / 'clips/'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        item = self.df.iloc[index]
        text = self.convert_to_integer(item.sentence)
        audio, _ = self.read(self.path / item.path)

        return torch.tensor(audio), torch.tensor(text)


mfcc = MFCC()


def ctc_collate_function(batch_chunk):

    x_batch, y_batch = list(), list()
    x_lengths, y_lengths = list(), list()

    for (x, y) in batch_chunk:
        # TODO add torchaudio.transform for some regulation
        # TODO optimize this loop. too slow

        x = mfcc(x)
        x = x.transpose(0, 1)

        x_batch.append(x)
        y_batch.append(y)

        x_lengths.append(x.shape[0] // 2)
        y_lengths.append(len(y))

    x_batch = nn.utils.rnn.pad_sequence(x_batch, batch_first=True) \
        .transpose(1, 2)
    # pad text with SPACE, with index of 43,
    # check table.csv and put it manually
    y_batch = nn.utils.rnn.pad_sequence(
        y_batch, batch_first=True, padding_value=43)

    return x_batch, y_batch, x_lengths, y_lengths


class SpeechCommand(Dataset, AudioUtility):
    # TODO
    def __init__(self) -> None:
        super().__init__()
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)
