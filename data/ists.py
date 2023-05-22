from dataclasses import dataclass
from typing import List

from datasets import load_dataset

from data import DifferenceDataset
from recognizers.utils import DifferenceSample


@dataclass
class ISTSSample(DifferenceSample):
    id: str = None


class ISTSDataset(DifferenceDataset):

    def __init__(self,
                 split: str = "train",
                 tgt_lang: str = "en",
                 ):
        super().__init__()
        assert split in {"train", "test"}
        self.split = split
        assert tgt_lang in {"en", "de", "es", "fr", "ja", "ko", "zh"}
        self.tgt_lang = tgt_lang
        self.dataset = load_dataset("ZurichNLP/rsd-ists-2016")[f"{split}_{tgt_lang}"]

    def get_samples(self) -> List[ISTSSample]:
        samples = []
        for sample in self.dataset:
            samples.append(ISTSSample(
                tokens_a=sample["tokens_a"],
                tokens_b=sample["tokens_b"],
                labels_a=sample["labels_a"],
                labels_b=sample["labels_b"],
                id=sample["id"],
            ))
        return samples

    def __str__(self):
        return f"ISTSDataset(en-{self.tgt_lang}, {self.split})"
