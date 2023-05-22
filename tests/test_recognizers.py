from unittest import TestCase

import torch

from recognizers import DiffAlign, DiffDel, DiffMask
from recognizers.diff_del import DiffDelWithReencode


class DiffAlignTestCase(TestCase):

    def setUp(self) -> None:
        self.recognizer = DiffAlign("google/bert_uncased_L-2_H-128_A-2")

    def test_str(self):
        print(self.recognizer)
        self.assertIn("DiffAlign", str(self.recognizer))
        self.assertIn("google/bert_uncased_L-2_H-128_A-2", str(self.recognizer))

    def test_predict(self):
        result = self.recognizer.predict(
            a="Chinese shares close higher Friday .",
            b="Chinese shares close lower Wednesday .",
        )
        self.assertEqual(('Chinese', 'shares', 'close', 'higher', 'Friday', '.'), result.tokens_a)
        self.assertEqual(('Chinese', 'shares', 'close', 'lower', 'Wednesday', '.'), result.tokens_b)
        self.assertEqual(6, len(result.labels_a))
        self.assertEqual(6, len(result.labels_b))
        self.assertIsInstance(result.labels_a[0], float)
        self.assertIsInstance(result.labels_b[0], float)

    def test_get_subword_ids_by_word(self):
        sentence = "This is a test sentence ."
        self.assertEqual([[1], [2], [3], [4], [5], [6]], self.recognizer._get_subwords_by_word(sentence))
        sentence = "This is a testtest sentence ."
        self.assertEqual([[1], [2], [3], [4, 5], [6], [7]], self.recognizer._get_subwords_by_word(sentence))

    def test_subword_labels_to_word_labels(self):
        subword_labels = torch.FloatTensor([0, 1, 0, 2, 1, 3])
        subwords_by_word = [[1], [2], [3, 4]]
        word_labels = self.recognizer._subword_labels_to_word_labels(subword_labels, subwords_by_word)
        self.assertListEqual([1, 0, 1.5], word_labels)


class DiffDelTestCase(TestCase):

    def setUp(self) -> None:
        self.recognizer = DiffDel("google/bert_uncased_L-2_H-128_A-2")

    def test_str(self):
        print(self.recognizer)
        self.assertIn("DiffDel", str(self.recognizer))
        self.assertIn("google/bert_uncased_L-2_H-128_A-2", str(self.recognizer))

    def test_predict(self):
        result = self.recognizer.predict(
            a="Chinese shares close higher Friday .",
            b="Chinese shares close lower Wednesday .",
        )
        self.assertEqual(('Chinese', 'shares', 'close', 'higher', 'Friday', '.'), result.tokens_a)
        self.assertEqual(('Chinese', 'shares', 'close', 'lower', 'Wednesday', '.'), result.tokens_b)
        self.assertEqual(6, len(result.labels_a))
        self.assertEqual(6, len(result.labels_b))
        self.assertIsInstance(result.labels_a[0], float)
        self.assertIsInstance(result.labels_b[0], float)

    def test_get_ngrams(self):
        subwords_by_word = [[1], [2], [3], [4, 5], [6], [7]]
        self.recognizer.min_n = 1
        self.recognizer.max_n = 1
        self.assertEqual([[1], [2], [3], [4], [5], [6], [7]], self.recognizer._get_ngrams(subwords_by_word))
        self.recognizer.min_n = 1
        self.recognizer.max_n = 2
        self.assertEqual([[1], [2], [3], [4], [5], [6], [7], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]], self.recognizer._get_ngrams(subwords_by_word))

    def test_get_ngrams_sentence_shorter_than_n(self):
        subwords_by_word = [[1], [2], [3], [4, 5], [6], [7]]
        self.recognizer.min_n = 8
        self.recognizer.max_n = 8
        self.assertEqual([[1, 2, 3, 4, 5, 6, 7]], self.recognizer._get_ngrams(subwords_by_word))

    def test_pool(self):
        token_embeddings = [
            [[1, 2], [3, 4], [5, 6]],
            [[1, 2], [3, 4], [5, 6]],
            [[1, 2], [3, 4], [5, 6]],
        ]
        mask = [
            [1, 1, 1],
            [1, 0, 0],
            [0, 0, 0],
        ]
        expected = [
            [9, 12],
            [1,  2],
            [0,  0],
        ]
        self.assertEqual(expected, self.recognizer._pool(torch.Tensor(token_embeddings), torch.Tensor(mask)).tolist())


class DiffDelWithReencodeTestCase(TestCase):

    def setUp(self) -> None:
        self.recognizer = DiffDelWithReencode("google/bert_uncased_L-2_H-128_A-2")

    def test_str(self):
        print(self.recognizer)
        self.assertIn("DiffDelWithReencode", str(self.recognizer))
        self.assertIn("google/bert_uncased_L-2_H-128_A-2", str(self.recognizer))

    def test_predict(self):
        result = self.recognizer.predict(
            a="Chinese shares close higher Friday .",
            b="Chinese shares close lower Wednesday .",
        )
        self.assertEqual(('Chinese', 'shares', 'close', 'higher', 'Friday', '.'), result.tokens_a)
        self.assertEqual(('Chinese', 'shares', 'close', 'lower', 'Wednesday', '.'), result.tokens_b)
        self.assertEqual(6, len(result.labels_a))
        self.assertEqual(6, len(result.labels_b))
        self.assertIsInstance(result.labels_a[0], float)
        self.assertIsInstance(result.labels_b[0], float)


class DiffMaskTestCase(TestCase):

    def setUp(self) -> None:
        self.recognizer = DiffMask("google/bert_uncased_L-2_H-128_A-2")

    def test_str(self):
        print(self.recognizer)
        self.assertIn("DiffMask", str(self.recognizer))
        self.assertIn("google/bert_uncased_L-2_H-128_A-2", str(self.recognizer))

    def test_predict(self):
        result = self.recognizer.predict(
            a="Chinese shares close higher Friday .",
            b="Chinese shares close lower Wednesday .",
        )
        self.assertEqual(('Chinese', 'shares', 'close', 'higher', 'Friday', '.'), result.tokens_a)
        self.assertEqual(('Chinese', 'shares', 'close', 'lower', 'Wednesday', '.'), result.tokens_b)
        self.assertEqual(6, len(result.labels_a))
        self.assertEqual(6, len(result.labels_b))
        self.assertIsInstance(result.labels_a[0], float)
        self.assertIsInstance(result.labels_b[0], float)
