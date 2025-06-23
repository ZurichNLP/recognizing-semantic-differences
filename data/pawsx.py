import string
from dataclasses import dataclass
from typing import List

from datasets import load_dataset

from data import DifferenceDataset
from recognizers.utils import DifferenceSample, tokenize


@dataclass
class PAWSXSample(DifferenceSample):
    id: str = None


class PAWSXDataset(DifferenceDataset):

    def __init__(self, language: str = "en", split: str = "validation"):
        super().__init__()
        assert language in {"en", "de", "es", "fr", "ja", "ko", "zh"}
        self.language = language
        assert split in {"train", "test", "validation"}
        self.split = split
        dataset = load_dataset("paws-x", language, split=self.split)
        # Only use sentence pairs that are paraphrases
        self.dataset = dataset.filter(lambda example: example["label"] == 1)

    def get_samples(self) -> List[PAWSXSample]:
        samples = []
        for row in self.dataset:
            # Exclude a small number of wrongly formatted samples
            # (https://github.com/google-research-datasets/paws/issues/15)
            if "NS" in {row["sentence1"].strip(), row["sentence2"].strip()}:
                continue

            tokens_a = tokenize(row["sentence1"])
            tokens_b = tokenize(row["sentence2"])

            # All tokens are labeled as "no difference"
            labels_a = [0. for _ in tokens_a]
            labels_b = [0. for _ in tokens_b]

            # Punctuation marks are ignored
            for i, token in enumerate(tokens_a):
                if token in string.punctuation:
                    labels_a[i] = -1
            for i, token in enumerate(tokens_b):
                if token in string.punctuation:
                    labels_b[i] = -1

            samples.append(PAWSXSample(
                tokens_a=tuple(tokens_a),
                tokens_b=tuple(tokens_b),
                labels_a=tuple(labels_a),
                labels_b=tuple(labels_b),
                id=str(row["id"]),
            ))
        return samples

    def __str__(self):
        return f"PAWS-X ({self.language}, {self.split})"


class CrosslingualPAWSXDataset(PAWSXDataset):

    def __init__(self, tgt_lang: str, split: str = "test"):
        super().__init__("en", split=split)
        self.tgt_dataset = PAWSXDataset(language=tgt_lang, split=split)

    def get_samples(self) -> List[PAWSXSample]:
        src_samples = super().get_samples()
        tgt_samples = self.tgt_dataset.get_samples()
        src_dict = {sample.id: sample for sample in src_samples}
        tgt_dict = {sample.id: sample for sample in tgt_samples}
        all_ids = set(src_dict.keys()) & set(tgt_dict.keys())
        samples = []
        for id in all_ids:
            src_sample = src_dict.get(id, None)
            tgt_sample = tgt_dict.get(id, None)

            # Only keep samples that are in both language versions of the dataset
            if src_sample is None or tgt_sample is None:
                continue

            samples.append(PAWSXSample(
                tokens_a=src_sample.tokens_a,
                tokens_b=tgt_sample.tokens_b,
                labels_a=src_sample.labels_a,
                labels_b=tuple(-1 for _ in tgt_sample.tokens_b),
                id=f"{src_sample.id}-forward",
            ))
            samples.append(PAWSXSample(
                tokens_a=src_sample.tokens_b,
                tokens_b=tgt_sample.tokens_a,
                labels_a=src_sample.labels_b,
                labels_b=tuple(-1 for _ in tgt_sample.tokens_a),
                id=f"{src_sample.id}-backward",
            ))
        return samples

    def __str__(self):
        return f"Crosslingual PAWS-X ({self.language}-{self.tgt_dataset.language}, {self.split})"
