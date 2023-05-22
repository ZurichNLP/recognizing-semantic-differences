from unittest import TestCase

from data.ists import ISTSDataset
from data.pawsx import PAWSXDataset, CrosslingualPAWSXDataset
from recognizers.utils import DifferenceSample


class ISTSDatasetTestCase(TestCase):

    def setUp(self) -> None:
        self.dataset = ISTSDataset()

    def test_get_samples(self):
        sample: DifferenceSample = self.dataset.get_samples()[0]
        self.assertEqual(len(sample.tokens_a), len(sample.labels_a))
        self.assertEqual(len(sample.tokens_b), len(sample.labels_b))
        print(sample)


class CrosslingualISTSDatasetTestCase(TestCase):

    def setUp(self) -> None:
        self.dataset = ISTSDataset(tgt_lang="de")

    def test_get_samples(self):
        samples = self.dataset.get_samples()
        sample = samples[0]
        self.assertEqual(len(sample.tokens_a), len(sample.labels_a))
        self.assertEqual(len(sample.tokens_b), len(sample.labels_b))
        self.assertSetEqual(set(sample.labels_b), {-1})
        print(samples[0])
        print(samples[1])


class PAWSXDatasetTestCase(TestCase):

    def setUp(self) -> None:
        self.dataset = PAWSXDataset()

    def test_get_samples(self):
        sample: DifferenceSample = self.dataset.get_samples()[0]
        self.assertEqual(len(sample.tokens_a), len(sample.labels_a))
        self.assertEqual(len(sample.tokens_b), len(sample.labels_b))
        print(sample)


class CrosslingualPAWSXDatasetTestCase(TestCase):

    def setUp(self) -> None:
        self.dataset = CrosslingualPAWSXDataset(
            tgt_lang="de",
        )

    def test_get_samples(self):
        samples = self.dataset.get_samples()
        sample = samples[0]
        self.assertEqual(len(sample.tokens_a), len(sample.labels_a))
        self.assertEqual(len(sample.tokens_b), len(sample.labels_b))
        self.assertSetEqual(set(sample.labels_b), {-1})
        print(samples[0])
        print(samples[1])
