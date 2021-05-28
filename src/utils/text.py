from typing import Dict, List, Text
import os
import csv
import torch
from persian import persian

class TableGenerator:
    """Generate Lookup-table
    """

    PERSIAN_ALPHABET = "ا ب پ ت ث ج چ ح خ د ذ ر ز ژ س ش ص ض ط ظ ع غ ف ق ک گ ل م ن و ه ی آ"
    ENGLISH_NUMBERS = "0 1 2 3 4 5 6 7 8 9"

    CHARS = [
        PERSIAN_ALPHABET,
        ENGLISH_NUMBERS
    ]

    def __init__(self) -> None:
        self.table = {}
        self.add_characters()
        self.add_symbols()

    @property
    def last_index(self):
        return len(self.table)

    def add_characters(self):
        index = self.last_index

        for sequence in self.CHARS:
            for char in sequence.split(" "):
                self.table[index] = char
                index += 1

    def add_symbols(self):
        # right now only <space> symbol will be here
        self.table[self.last_index] = " "

class Filter:
    def __init__(self, lookup_table: Dict[int, str], default_replace=" ") -> None:
        self._lookup_table = lookup_table
        self.default_replace = default_replace

    @property
    def chars(self):
        """return list of all characters mapped in lookup-table
        """
        return self._lookup_table.values()

    def clear_symbols(self, text):
        """
        """
        filter = lambda char: char if char in self.chars else self.default_replace

        text_as_list = list(text)
        text_as_list =  map(filter, text_as_list)
        return "".join(list(text_as_list))

    def clear_space_duplication(self, text):
        return " ".join(text.split())

    def replace_numbers(self, text):
        """replace arabic and persian numebrs with english ones.
        """
        return persian.convert_fa_numbers(text)

    def apply_filters(self, text):
        text = self.clear_symbols(text)
        text = self.clear_space_duplication(text)
        text = self.replace_numbers(text)
        return text

class Decoder:
    """ CTC output decoder based on Beam and Greedy search.
    """

    def greedy(self, probs: torch.Tensor) -> List[int]:
        result = torch.argmax(probs, dim=1)
        return list(map(int, result))

class TextUtility(Filter, Decoder):
    """Turn raw Text into numberical representation in character-level and vice versa.
    """
    _lookup_table = {}

    def __init__(self, config) -> None:
        if "table_path" in config.text_utility:
            self.load_table(config.text_utility.table_path)
        else:
            gen = TableGenerator()
            self._lookup_table = gen.table

        super().__init__(self._lookup_table, default_replace=config.text_utility.replacer)

    def load_table(self, table_path):
        # load character-level lookup table 
        # from .csv file
        assert os.path.isfile(table_path), ".csv file for lookup-tabe does not exitst, (text_utility/table_path)"
        with open(table_path, encoding='utf-8', newline="") as file:
            csv_table = csv.reader(file, delimiter=",")

            for row in csv_table:
                index = row[0]
                char = row[1]
                self._lookup_table[int(index)] = char

    def export_table(self, destination: str):
        """export look-up table into .csv file
        """
        with open(destination, 'w', encoding='utf-8', newline="") as file:
            writer = csv.writer(file)

            for key in self._lookup_table:
                value = self._lookup_table[key]
                writer.writerow((int(key), value))

    @property
    def reversed_lookup_table(self):
        """reverse index->char to char->index.
        """
        return {value:key for key, value in self._lookup_table.items()}

    @property
    def table_length(self):
        return len(self._lookup_table)

    @property
    def blank_id(self):
        # if table length is 27, 
        # blank_id would be 28 for CTC model
        return self.table_length + 1

    def convert_to_integer(self, text):
        text = self.apply_filters(text)
        return [int(self.reversed_lookup_table[char]) for char in text]

    def convert_to_text(self, arr: List[int]) -> Text:
        print(self._lookup_table)
        return "".join(self._lookup_table[index] for index in arr)

