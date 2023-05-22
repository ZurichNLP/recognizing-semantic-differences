import itertools
from typing import List, Union

import torch
import transformers
from transformers import FeatureExtractionPipeline, Pipeline

from recognizers.base import DifferenceRecognizer
from recognizers.utils import DifferenceSample

Ngram = List[int]  # A span of subword indices


class FeatureExtractionRecognizer(DifferenceRecognizer):

    def __init__(self,
                 model_name_or_path: str = None,
                 pipeline: Union[FeatureExtractionPipeline, Pipeline] = None,
                 layer: int = -1,
                 batch_size: int = 16,
                 ):
        assert model_name_or_path is not None or pipeline is not None
        if pipeline is None:
            pipeline = transformers.pipeline(
                model=model_name_or_path,
                task="feature-extraction",
            )
        self.pipeline = pipeline
        self.layer = layer
        self.batch_size = batch_size

    def encode_batch(self, sentences: List[str], **kwargs) -> torch.Tensor:
        model_inputs = self.pipeline.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        model_inputs = model_inputs.to(self.pipeline.device)
        outputs = self.pipeline.model(**model_inputs, output_hidden_states=True, **kwargs)
        return outputs.hidden_states[self.layer]

    def predict(self,
                a: str,
                b: str,
                **kwargs,
                ) -> DifferenceSample:
        return self.predict_all([a], [b], **kwargs)[0]

    def predict_all(self,
                    a: List[str],
                    b: List[str],
                    **kwargs,
                    ) -> List[DifferenceSample]:
        samples = []
        for i in range(0, len(a), self.batch_size):
            samples.extend(self._predict_all(
                a[i:i + self.batch_size],
                b[i:i + self.batch_size],
                **kwargs,
            ))
        return samples

    @torch.no_grad()
    def _predict_all(self,
                    a: List[str],
                    b: List[str],
                    **kwargs,
                    ) -> List[DifferenceSample]:
        raise NotImplementedError

    def _pool(self, token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param token_embeddings: batch x seq_len x dim
        :param mask: batch x seq_len; 1 if token should be included in the pooling
        :return: batch x dim
        Do only sum and do not divide by the number of tokens because cosine similarity is length-invariant.
        """
        return torch.sum(token_embeddings * mask.unsqueeze(-1), dim=1)

    def _get_subwords_by_word(self, sentence: str) -> List[Ngram]:
        """
        :return: For each word in the sentence, the positions of the subwords that make up the word.
        """
        batch_encoding = self.pipeline.tokenizer(
            sentence,
            padding=True,
            truncation=True,
        )
        subword_ids: List[List[int]] = []

        for subword_idx in range(len(batch_encoding.encodings[0].word_ids)):
            if batch_encoding.encodings[0].word_ids[subword_idx] is None:  # Special token
                continue
            char_idx = batch_encoding.encodings[0].offsets[subword_idx][0]
            if isinstance(self.pipeline.tokenizer, transformers.XLMRobertaTokenizerFast) or \
                    isinstance(self.pipeline.tokenizer, transformers.XLMRobertaTokenizer):
                token = batch_encoding.encodings[0].tokens[subword_idx]
                is_tail = not token.startswith("▁") and token not in self.pipeline.tokenizer.all_special_tokens
            elif isinstance(self.pipeline.tokenizer, transformers.RobertaTokenizerFast) or \
                    isinstance(self.pipeline.tokenizer, transformers.RobertaTokenizer):
                token = batch_encoding.encodings[0].tokens[subword_idx]
                is_tail = not token.startswith("Ġ") and token not in self.pipeline.tokenizer.all_special_tokens
            else:
                is_tail = char_idx > 0 and char_idx == batch_encoding.encodings[0].offsets[subword_idx - 1][1]
            if is_tail and len(subword_ids) > 0:
                subword_ids[-1].append(subword_idx)
            else:
                subword_ids.append([subword_idx])
        return subword_ids

    def _get_ngrams(self, subwords_by_word: List[Ngram]) -> List[Ngram]:
        """
        :return: For each subword ngram in the sentence, the positions of the subwords that make up the ngram.
        """
        subwords = list(itertools.chain.from_iterable(subwords_by_word))
        # Always return at least one ngram (reduce n if necessary)
        min_n = min(self.min_n, len(subwords))
        ngrams = []
        for n in range(min_n, self.max_n + 1):
            for i in range(len(subwords) - n + 1):
                ngrams.append(subwords[i:i + n])
        return ngrams

    def _subword_labels_to_word_labels(self, subword_labels: torch.Tensor, subwords_by_words: List[Ngram]) -> List[float]:
        """
        :param subword_labels: num_subwords
        :param subwords_by_words: num_words x num_subwords
        :return: num_words
        """
        labels = []
        for subword_indices in subwords_by_words:
            label = subword_labels[subword_indices].mean().item()
            labels.append(label)
        return labels
