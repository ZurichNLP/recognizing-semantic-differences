import random
from typing import Dict, Tuple
from unittest import TestCase

import numpy as np

from data.ists import ISTSDataset
from data.pawsx import PAWSXDataset
from experiments.benchmark import DifferenceRecognitionBenchmark
from recognizers.base import DifferenceRecognizer
from recognizers.utils import DifferenceSample


class DifferenceRecognitionBenchmarkTestCase(TestCase):

    def setUp(self) -> None:
        self.positive_dataset = ISTSDataset()
        self.negative_dataset = PAWSXDataset()
        self.benchmark = DifferenceRecognitionBenchmark(
            positive_dataset=self.positive_dataset,
        )

    def test_str(self):
        print(self.benchmark)

    def test_documents(self):
        document = self.benchmark.documents[0]
        self.assertEqual(len(document.tokens_a), len(document.labels_a))
        self.assertEqual(len(document.tokens_b), len(document.labels_b))
        print(document)

    def test_negative_examples(self):
        benchmark = DifferenceRecognitionBenchmark(
            positive_dataset=self.positive_dataset,
            negative_dataset=self.negative_dataset,
            positive_ratio=0.5,
        )
        document = benchmark.documents[0]
        self.assertEqual(len(document.tokens_a), len(document.labels_a))
        self.assertEqual(len(document.tokens_b), len(document.labels_b))
        print(document)

    def test_long_documents(self):
        benchmark = DifferenceRecognitionBenchmark(
            positive_dataset=self.positive_dataset,
            num_sentences_per_document=5,
        )
        document = benchmark.documents[0]
        self.assertEqual(len(document.tokens_a), len(document.labels_a))
        self.assertEqual(len(document.tokens_b), len(document.labels_b))
        print(document)

    def test_permutations(self):
        benchmark = DifferenceRecognitionBenchmark(
            positive_dataset=self.positive_dataset,
            num_sentences_per_document=5,
            num_inversions=2,
        )
        document = benchmark.documents[0]
        self.assertEqual(len(document.tokens_a), len(document.labels_a))
        self.assertEqual(len(document.tokens_b), len(document.labels_b))
        print(document)

    def test_statistics(self):
        print(self.benchmark.num_document_pairs)
        print(self.benchmark.num_tokens)
        print(self.benchmark.num_labels_lt_05)
        print(self.benchmark.num_labels_gte_05)
        print(self.benchmark.num_unlabeled_tokens)
        self.assertGreater(self.benchmark.num_tokens, self.benchmark.num_document_pairs)
        self.assertEqual(self.benchmark.num_labels_lt_05 + self.benchmark.num_labels_gte_05 + self.benchmark.num_unlabeled_tokens, self.benchmark.num_tokens)

    def test_evaluate_random(self):
        class RandomRecognizer(DifferenceRecognizer):
            def predict(self, a: str, b: str, *args, **kwargs):
                return DifferenceSample(
                    tokens_a=tuple(a.split()),
                    tokens_b=tuple(b.split()),
                    labels_a=tuple(random.random() for _ in a.split()),
                    labels_b=tuple(random.random() for _ in b.split()),
                )
        recognizer = RandomRecognizer()
        results = []
        for _ in range(100):
            result = self.benchmark.evaluate(recognizer)
            results.append(result)
        mean_result = np.mean([result.spearman for result in results])
        self.assertAlmostEqual(0, mean_result, places=2)

    def test_evaluate_oracle(self):
        class OracleRecognizer(DifferenceRecognizer):

            def __init__(self, benchmark: DifferenceRecognitionBenchmark):
                self.benchmark = benchmark
                self.gold_labels_a: Dict[str, Tuple[float, ...]] = {}
                self.gold_labels_b: Dict[str, Tuple[float, ...]] = {}
                for sample in self.benchmark.documents:
                    self.gold_labels_a[" ".join(sample.tokens_a) + " ".join(sample.tokens_b)] = sample.labels_a
                    self.gold_labels_b[" ".join(sample.tokens_a) + " ".join(sample.tokens_b)] = sample.labels_b

            def predict(self, a: str, b: str, *args, **kwargs):
                return DifferenceSample(
                    tokens_a=tuple(a.split()),
                    tokens_b=tuple(b.split()),
                    labels_a=self.gold_labels_a[a + b],
                    labels_b=self.gold_labels_b[a + b],
                )

        recognizer = OracleRecognizer(self.benchmark)
        result = self.benchmark.evaluate(recognizer)
        self.assertEqual(1, result.spearman)
