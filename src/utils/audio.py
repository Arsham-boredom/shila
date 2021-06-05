from typing import Tuple
import librosa
import numpy as np

#TODO
class Record:
    pass

class AudioUtility(Record):
    __sound_rate__ = 22050

    def read(self, file_path):
        # read audio file 
        return librosa.load(file_path, sr=self.__sound_rate__)

    def mfcc(self, wav: np.array) -> Tuple[np.array, int]:
        return librosa.feature.mfcc(wav, sr=self.__sound_rate__) 