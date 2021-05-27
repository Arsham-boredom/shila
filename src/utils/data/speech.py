from typing import Tuple
import torch
from torch import Tensor, nn
from pandas import DataFrame
from torch.utils.data import Dataset
from src.utils.text import TextUtility
from src.utils.audio import AudioUtility

class CommonVoice(Dataset, TextUtility, AudioUtility):

    def __init__(self, df, config) -> None:
        super().__init__()
        super(TextUtility, self).__init__(config=config)
        # common voice .csv file
        self.df: DataFrame = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, int, int]:
        item = self.df.iloc[index]
        text = self.convert_to_integer(item.sentence)
        audio = self.read(item.path)

        return torch.tensor(audio), torch.tensor(text), len(audio), len(text)

def collate_function(batch):
    #TODO add torchaudio.transform for some regulation
    x_batch, y_batch , x_len, y_len = batch
    x_batch = nn.utils.rnn.pad_sequence(x_batch, batch_first=True)

    # 43 is the index of SPACE in char table, look at table.csv I put it here manually
    # this will pad text labels with SPACE
    y_batch = nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value=43) 

    return x_batch, y_batch, x_len, y_len