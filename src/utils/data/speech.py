from typing import Tuple
from numpy import pad
import torch
from torch import Tensor, nn
from pandas import DataFrame
from torch.utils.data import Dataset
from src.utils.text import TextUtility
from src.utils.audio import AudioUtility

DIRECORY_NAME = "cv-corpus-6.1-2020-12-11"

def get_complete_path(filename):
  return "{}/fa/clips/{}".format(DIRECORY_NAME, filename)

class CommonVoice(Dataset, TextUtility, AudioUtility):

    def __init__(self, df, config) -> None:
        super().__init__(config=config)
        # common voice .csv file
        self.df: DataFrame = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        item = self.df.iloc[index]
        text = self.convert_to_integer(item.sentence)
        audio, _ = self.read(get_complete_path(item.path))

        return torch.tensor(audio), torch.tensor(text)

T = lambda t: t.view((t.size(0), 1, *t.size()[1:]))

def ctc_collate_function(batch_chunk):

    x_batch, y_batch = list(), list()
    x_lengths, y_lengths = list(), list()

    for (x, y) in batch_chunk:
        #TODO add torchaudio.transform for some regulation
        #TODO optimize this loop. too slow

        x_batch.append(x)
        y_batch.append(y)

        x_lengths.append(len(x))
        y_lengths.append(len(y))

    # pad wav arrays with 0, silence
    x_batch = nn.utils.rnn.pad_sequence(x_batch, batch_first=True)
    x_batch = T(x_batch)
    # pad text with SPACE, with index of 43, check table.csv and put it manually
    y_batch = nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value=43)

    return x_batch, y_batch, x_lengths, y_lengths

class SpeechCommand(Dataset, AudioUtility):
    #TODO
    def __init__(self) -> None:
        super().__init__()
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)