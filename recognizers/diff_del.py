import itertools
from copy import deepcopy
from typing import Union, List

import torch
from transformers import Pipeline, FeatureExtractionPipeline

from recognizers.feature_based import FeatureExtractionRecognizer, Ngram
from recognizers.utils import DifferenceSample, pairwise_cos_sim, cos_sim


class DiffDel(FeatureExtractionRecognizer):

    def __init__(self,
                 model_name_or_path: str = None,
                 pipeline: Union[FeatureExtractionPipeline, Pipeline] = None,
                 layer: int = -1,
                 batch_size: int = 16,
                 min_n: int = 1,
                 max_n: int = 1,  # Inclusive
                 ):
        super().__init__(model_name_or_path, pipeline, layer, batch_size)
        assert min_n <= max_n
        self.min_n = min_n
        self.max_n = max_n

    def __str__(self):
        return f"DiffDel(model={self.pipeline.model.name_or_path}, layer={self.layer}, " \
               f"min_n={self.min_n}, max_n={self.max_n})"

    @torch.no_grad()
    def _predict_all(self,
                    a: List[str],
                    b: List[str],
                    **kwargs,
                    ) -> List[DifferenceSample]:
        outputs_a = self.encode_batch(a, **kwargs)
        outputs_b = self.encode_batch(b, **kwargs)
        subwords_by_words_a = [self._get_subwords_by_word(sentence) for sentence in a]
        subwords_by_words_b = [self._get_subwords_by_word(sentence) for sentence in b]
        ngrams_a = [self._get_ngrams(subwords_by_word) for subwords_by_word in subwords_by_words_a]
        ngrams_b = [self._get_ngrams(subwords_by_word) for subwords_by_word in subwords_by_words_b]
        sentence_embeddings_a = self._get_full_sentence_embeddings(outputs_a, [list(itertools.chain.from_iterable(subwords)) for subwords in subwords_by_words_a])
        sentence_embeddings_b = self._get_full_sentence_embeddings(outputs_b, [list(itertools.chain.from_iterable(subwords)) for subwords in subwords_by_words_b])
        full_similarities = pairwise_cos_sim(sentence_embeddings_a, sentence_embeddings_b)

        all_labels_a = []
        all_labels_b = []
        for i in range(len(a)):
            partial_embeddings_a = self._get_partial_sentence_embeddings_for_sample(outputs_a[i], ngrams_a[i])
            partial_embeddings_b = self._get_partial_sentence_embeddings_for_sample(outputs_b[i], ngrams_b[i])
            partial_similarities_a = cos_sim(partial_embeddings_a, sentence_embeddings_b[i].unsqueeze(0)).squeeze(1)
            partial_similarities_b = cos_sim(partial_embeddings_b, sentence_embeddings_a[i].unsqueeze(0)).squeeze(1)
            ngram_labels_a = (partial_similarities_a - full_similarities[i] + 1) / 2
            ngram_labels_b = (partial_similarities_b - full_similarities[i] + 1) / 2
            subword_labels_a = self._distribute_ngram_labels_to_subwords(ngram_labels_a, ngrams_a[i])
            subword_labels_b = self._distribute_ngram_labels_to_subwords(ngram_labels_b, ngrams_b[i])
            labels_a = self._subword_labels_to_word_labels(subword_labels_a, subwords_by_words_a[i])
            labels_b = self._subword_labels_to_word_labels(subword_labels_b, subwords_by_words_b[i])
            all_labels_a.append(labels_a)
            all_labels_b.append(labels_b)

        samples = []
        for i in range(len(a)):
            samples.append(DifferenceSample(
                tokens_a=tuple(a[i].split()),
                tokens_b=tuple(b[i].split()),
                labels_a=tuple(all_labels_a[i]),
                labels_b=tuple(all_labels_b[i]),
            ))
        return samples

    def _get_full_sentence_embeddings(self, token_embeddings: torch.Tensor, include_subwords: List[List[int]]) -> torch.Tensor:
        """
        :param token_embeddings: batch x seq_len x dim
        :param include_subwords: batch x num_subwords
        :return: A tensor of shape batch x dim
        """
        pool_mask = torch.zeros(token_embeddings.shape[0], token_embeddings.shape[1], device=token_embeddings.device)
        for i, subword_indices in enumerate(include_subwords):
            pool_mask[i, subword_indices] = 1
        sentence_embeddings = self._pool(token_embeddings, pool_mask)
        return sentence_embeddings

    def _get_partial_sentence_embeddings_for_sample(self, token_embeddings: torch.Tensor, ngrams: List[Ngram]) -> torch.Tensor:
        """
        :param token_embeddings: seq_len x dim
        :param ngrams: num_ngrams x n
        :return: A tensor of shape num_ngrams x dim
        """
        pool_mask = torch.zeros(len(ngrams), token_embeddings.shape[0], device=token_embeddings.device)
        pool_mask[:, list(itertools.chain.from_iterable(ngrams))] = 1
        for i, subword_indices in enumerate(ngrams):
            pool_mask[i, subword_indices] = 0
        partial_embeddings = self._pool(token_embeddings.unsqueeze(0).repeat(len(ngrams), 1, 1), pool_mask)
        return partial_embeddings

    def _distribute_ngram_labels_to_subwords(self, ngram_labels: torch.Tensor, ngrams: List[Ngram]) -> torch.Tensor:
        """
        :param ngram_labels: num_ngrams
        :param ngrams: num_ngrams x n
        :return: num_subwords
        """
        max_subword_idx = max(itertools.chain.from_iterable(ngrams))
        subword_contributions = torch.zeros(max_subword_idx + 1, device=ngram_labels.device)
        contribution_count = torch.zeros(max_subword_idx + 1, device=ngram_labels.device)
        for i, ngram in enumerate(ngrams):
            subword_contributions[ngram] += ngram_labels[i] / len(ngram)
            contribution_count[ngram] += 1 / len(ngram)
        subword_contributions /= contribution_count
        return subword_contributions


class DiffDelWithReencode(FeatureExtractionRecognizer):
    """
    Version of DiffDel that encodes the partial sentences from scratch (instead of encoding the full sentence once and
    then excluding hidden states from the mean)
    """

    def __init__(self,
                 model_name_or_path: str = None,
                 pipeline: Union[FeatureExtractionPipeline, Pipeline] = None,
                 layer: int = -1,
                 batch_size: int = 16,
                 ):
        super().__init__(model_name_or_path, pipeline, layer, batch_size)

    def __str__(self):
        return f"DiffDelWithReencode(model={self.pipeline.model.name_or_path}, layer={self.layer})"

    @torch.no_grad()
    def _predict_all(self,
                    a: List[str],
                    b: List[str],
                    **kwargs,
                    ) -> List[DifferenceSample]:
        a_words = [sentence.split() for sentence in a]
        b_words = [sentence.split() for sentence in b]
        a_words_partial = []
        b_words_partial = []
        for words in a_words:
            for i, word in enumerate(words):
                partial = deepcopy(words)
                del partial[i]
                a_words_partial.append(partial)
        for words in b_words:
            for i, word in enumerate(words):
                partial = deepcopy(words)
                del partial[i]
                b_words_partial.append(partial)
        a_partial = [" ".join([word for word in words if word]) for words in a_words_partial]
        b_partial = [" ".join([word for word in words if word]) for words in b_words_partial]
        a_num_partial = [len(words) for words in a_words]
        b_num_partial = [len(words) for words in b_words]
        a_embedding_full = self._encode_and_pool(a, **kwargs)
        b_embedding_full = self._encode_and_pool(b, **kwargs)
        a_embeddings_partial = []
        b_embeddings_partial = []
        for i in range(0, len(a_partial), self.batch_size):
            a_embeddings_partial_batch = self._encode_and_pool(a_partial[i:i + self.batch_size], **kwargs)
            a_embeddings_partial.append(a_embeddings_partial_batch)
        for i in range(0, len(b_partial), self.batch_size):
            b_embeddings_partial_batch = self._encode_and_pool(b_partial[i:i + self.batch_size], **kwargs)
            b_embeddings_partial.append(b_embeddings_partial_batch)
        a_embeddings_partial = torch.cat(a_embeddings_partial, dim=0)
        b_embeddings_partial = torch.cat(b_embeddings_partial, dim=0)

        labels_a = []
        labels_b = []
        similarity_full = pairwise_cos_sim(a_embedding_full, b_embedding_full)
        for i in range(len(a)):
            a_embeddings_partial_i = a_embeddings_partial[sum(a_num_partial[:i]):sum(a_num_partial[:i + 1])]
            similarities_partial = pairwise_cos_sim(a_embeddings_partial_i, b_embedding_full[i].unsqueeze(0)).squeeze(0)
            labels = (similarities_partial - similarity_full[i] + 1) / 2
            labels = labels.detach().cpu().tolist()
            if isinstance(labels, float):
                labels = [labels]
            assert len(labels) == len(a_words[i])
            labels_a.append(labels)
        for i in range(len(b)):
            b_embeddings_partial_i = b_embeddings_partial[sum(b_num_partial[:i]):sum(b_num_partial[:i + 1])]
            similarities_partial = pairwise_cos_sim(b_embeddings_partial_i, a_embedding_full[i].unsqueeze(0)).squeeze(0)
            labels = (similarities_partial - similarity_full[i] + 1) / 2
            labels = labels.detach().cpu().tolist()
            if isinstance(labels, float):
                labels = [labels]
            assert len(labels) == len(b_words[i])
            labels_b.append(labels)

        samples = []
        for i in range(len(a)):
            samples.append(DifferenceSample(
                tokens_a=tuple(a_words[i]),
                tokens_b=tuple(b_words[i]),
                labels_a=tuple(labels_a[i]),
                labels_b=tuple(labels_b[i]),
            ))
        return samples

    def _encode_and_pool(self, sentences: List[str], **kwargs) -> torch.Tensor:
        model_inputs = self.pipeline.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        model_inputs = model_inputs.to(self.pipeline.device)
        outputs = self.pipeline.model(**model_inputs, output_hidden_states=True, **kwargs)
        if self.layer == "mean":
            token_embeddings = torch.stack(outputs.hidden_states, dim=0).mean(dim=0)
        else:
            assert isinstance(self.layer, int)
            token_embeddings = outputs.hidden_states[self.layer]
        mask = model_inputs["attention_mask"]
        sentence_embeddings = torch.sum(token_embeddings * mask.unsqueeze(-1), dim=1)
        return sentence_embeddings
