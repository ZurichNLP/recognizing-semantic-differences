import itertools
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List

import permpy
from scipy.stats import spearmanr

from data import DifferenceDataset
from recognizers.base import DifferenceRecognizer
from recognizers.utils import DifferenceSample


@dataclass
class DifferenceRecognitionResult:
    spearman: float

    def __repr__(self):
        return f"{self.spearman:.3f}"


class DifferenceRecognitionBenchmark:

    def __init__(self,
                 positive_dataset: DifferenceDataset,
                 negative_dataset: DifferenceDataset = None,
                 positive_ratio: float = 1.0,
                 num_sentences_per_document: int = 1,
                 num_inversions: int = 0,
                 seed: int = 42,
                 ):
        self.positive_ratio = positive_ratio
        assert 0 <= self.positive_ratio <= 1
        if positive_ratio < 1:
            assert negative_dataset is not None
        self.num_sentences_per_document = num_sentences_per_document
        self.num_inversions = num_inversions
        assert 0 <= self.num_inversions <= self.num_sentences_per_document * (self.num_sentences_per_document - 1) / 2
        self.seed = seed
        self.random = random.Random(self.seed)
        self.positive_dataset = positive_dataset
        self.negative_dataset = negative_dataset
        self._load_sentences()
        self._build_documents()

    def _load_sentences(self):
        positive_sentences = self.positive_dataset.get_samples()
        self.sentences = []
        self.sentences += positive_sentences
        if self.negative_dataset is not None:
            negative_sentences = self.negative_dataset.get_samples()
            num_sentences = int(len(positive_sentences) / self.positive_ratio)
            num_negatives = num_sentences - len(positive_sentences)
            self.sentences += negative_sentences[:num_negatives]
            if num_negatives > len(negative_sentences):
                for _ in range(num_negatives - len(negative_sentences)):
                    self.sentences.append(deepcopy(self.random.choice(negative_sentences)))
        self.random.shuffle(self.sentences)

    def _build_documents(self):
        self.documents: List[DifferenceSample] = []
        num_documents = math.floor(len(self.sentences) / self.num_sentences_per_document)
        if self.num_inversions > 0:
            self.permutations = [p for p in permpy.PermSet.all(self.num_sentences_per_document) if
                                 p.num_inversions() == self.num_inversions]
        else:
            self.permutations = None
        for i in range(num_documents):
            sentences = self.sentences[i * self.num_sentences_per_document:(i + 1) * self.num_sentences_per_document]
            document = DifferenceSample(
                tokens_a=tuple(itertools.chain.from_iterable([sentence.tokens_a for sentence in sentences])),
                tokens_b=tuple(itertools.chain.from_iterable([sentence.tokens_b for sentence in sentences])),
                labels_a=tuple(itertools.chain.from_iterable([sentence.labels_a for sentence in sentences])),
                labels_b=tuple(itertools.chain.from_iterable([sentence.labels_b for sentence in sentences])),
            )
            if self.permutations is not None:
                permutation = self.random.choice(self.permutations)
                sentences = [sentences[i] for i in permutation]  # Apply permutation
                document.tokens_b = tuple(itertools.chain.from_iterable([sentence.tokens_b for sentence in sentences]))
                document.labels_b = tuple(itertools.chain.from_iterable([sentence.labels_b for sentence in sentences]))
            self.documents.append(document)

    def __str__(self):
        return f"DifferenceRecognitionBenchmark(pos={self.positive_dataset}, neg={self.negative_dataset}, " \
                f"positive_ratio={self.positive_ratio}, num_sentences_per_document={self.num_sentences_per_document}, " \
                f"num_inversions={self.num_inversions}, seed={self.seed}"

    def evaluate(self,
                 recognizer: DifferenceRecognizer,
                 predict_kwargs: Dict = None,
                 ) -> DifferenceRecognitionResult:
        if predict_kwargs is None:
            predict_kwargs = {}
        predictions = recognizer.predict_all(
            a=[" ".join(sample.tokens_a) for sample in self.documents],
            b=[" ".join(sample.tokens_b) for sample in self.documents],
            **predict_kwargs,
        )
        assert len(predictions) == len(self.documents)
        gold_labels = []
        predicted_labels = []
        for sample, prediction in zip(self.documents, predictions):
            sample_gold_labels = list(sample.labels_a)
            sample_predicted_labels = list(prediction.labels_a)
            if set(sample.labels_b) != {-1}:
                sample_gold_labels += list(sample.labels_b)
                sample_predicted_labels += list(prediction.labels_b)
            assert len(sample_gold_labels) == len(sample_predicted_labels), f"Differing number of labels in sample {sample}: {len(sample_gold_labels)} vs. {len(sample_predicted_labels)}"
            sample_predicted_labels = [label for label, gold_label in zip(sample_predicted_labels, sample_gold_labels) if gold_label != -1]
            sample_gold_labels = [label for label in sample_gold_labels if label != -1]
            gold_labels += sample_gold_labels
            predicted_labels += sample_predicted_labels
        assert len(gold_labels) == len(predicted_labels)
        spearman = spearmanr(
            a=gold_labels,
            b=predicted_labels,
        ).correlation
        return DifferenceRecognitionResult(
            spearman=spearman,
        )

    @property
    def num_document_pairs(self) -> int:
        return len(self.documents)

    @property
    def num_tokens(self) -> int:
        num_tokens = 0
        for document in self.documents:
            num_tokens += len(document.tokens_a)
            num_tokens += len(document.tokens_b)
        return num_tokens

    @property
    def num_labels_lt_05(self) -> int:
        num_labels = 0
        for document in self.documents:
            num_labels += len([label for label in document.labels_a if 0 <= label < 0.5])
            num_labels += len([label for label in document.labels_b if 0 <= label < 0.5])
        return num_labels

    @property
    def num_labels_gte_05(self) -> int:
        num_labels = 0
        for document in self.documents:
            num_labels += len([label for label in document.labels_a if label >= 0.5])
            num_labels += len([label for label in document.labels_b if label >= 0.5])
        return num_labels

    @property
    def num_unlabeled_tokens(self) -> int:
        num_unlabeled_tokens = 0
        for document in self.documents:
            num_unlabeled_tokens += len([label for label in document.labels_a if label == -1])
            num_unlabeled_tokens += len([label for label in document.labels_b if label == -1])
        return num_unlabeled_tokens
