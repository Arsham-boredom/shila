from typing import Dict, Text
from omegaconf import OmegaConf
import os
import csv

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

    @property
    def last_index(self):
        return len(self.table)

    def add_characters(self):
        index = self.last_index

        for sequence in self.CHARS:
            for char in sequence.split(" "):
                index += 1
                self.table[index] = char

        # add blank character
        self.table[index+1] = "<blank>"

class Filter:
    def __init__(self, lookup_table: Dict[int, str], default_replace=" ") -> None:
        self._lookup_table = lookup_table
        self.default_replace = default_replace

    @property
    def chars(self):
        """return list of all characters mapped in lookup-table
        """
        return [value for key, value in self._lookup_table]

    def clear_symbols(self, text):
        """
        """
        filter = lambda char: char if char in self.chars else self.default_replace

        text_as_list = list(text)
        text_as_list =  map(filter, text_as_list)
        return list(text_as_list)

    def apply_filters(self, text):
        return self.clear_symbols(text)

class TextUtility(Filter):
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
                self._lookup_table[char] = index

    def export_table(self, destination: str):
        """export look-up table into .csv file
        """
        with open(destination, 'w', encoding='utf-8', newline="") as file:
            writer = csv.writer(file)

            for key in self._lookup_table:
                value = self._lookup_table[key]
                writer.writerow((key, value))

    @property
    def reversed_lookup_table(self):
        """reverse index->char to char->index.
        """
        return {value:key for key, value in self._lookup_table.items()}

    @property
    def blank_id(self):
        # if table length is 27, 
        # blank_id would be 28 for CTC model
        return len(self._lookup_table) + 1

    def convert_to_integer(self, text):
        text = self.apply_filters(text)
        return [self.reversed_lookup_table[char] for char in text]