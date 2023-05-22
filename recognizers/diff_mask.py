import itertools
from copy import deepcopy
from typing import List, Union

import torch
import transformers
from tqdm import tqdm
from transformers import FillMaskPipeline, Pipeline
from transformers.modeling_outputs import MaskedLMOutput

from recognizers.base import DifferenceRecognizer
from recognizers.feature_based import Ngram
from recognizers.utils import DifferenceSample


class DiffMask(DifferenceRecognizer):

    def __init__(self,
                 model_name_or_path: str = None,
                 pipeline: Union[FillMaskPipeline, Pipeline] = None,
                 batch_size: int = 8,
                 ):
        assert model_name_or_path is not None or pipeline is not None
        if pipeline is None:
            pipeline = transformers.pipeline(
                model=model_name_or_path,
                task="fill-mask",
            )
        self.pipeline = pipeline
        self.batch_size = batch_size

    def __str__(self):
        return f"DiffMask(model={self.pipeline.model.name_or_path})"

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
        all_samples = []
        batch_size = int(self.batch_size / 2)  # Every sentence needs to be encoded twice
        for i in tqdm(list(range(0, len(a), batch_size))):
            samples = self._predict_all(
                a[i:i + batch_size],
                b[i:i + batch_size],
                **kwargs,
            )
            all_samples.extend(samples)
        return all_samples

    @torch.no_grad()
    def _predict_all(self,
                    a: List[str],
                    b: List[str],
                    **kwargs,
                    ) -> List[DifferenceSample]:
        difference_a = self._predict_sentence(a, b, **kwargs)
        difference_b = self._predict_sentence(b, a, **kwargs)
        assert len(difference_a) == len(difference_b) == len(a)
        samples = []
        for i in range(len(a)):
            sample = DifferenceSample(
                tokens_a=difference_a[i].tokens_a,
                tokens_b=difference_b[i].tokens_a,
                labels_a=difference_a[i].labels_a,
                labels_b=difference_b[i].labels_a,
            )
            samples.append(sample)
        return samples

    def _predict_sentence(self,
                          a: List[str],
                          b: List[str],
                          **kwargs,
                          ) -> List[DifferenceSample]:
        subwords_by_words_a = [self._get_subwords_by_word(sentence) for sentence in a]
        a_without = deepcopy(a)
        a_with = [f"{b[i]} {self.pipeline.tokenizer.sep_token} {a[i]}" for i in range(len(a))]
        encoding_without = self.pipeline.tokenizer(
            a_without,
            padding="max_length",
            truncation=True,
            max_length=self.pipeline.model.config.max_position_embeddings,
            return_tensors="pt",
        ).to(self.pipeline.device)
        encoding_with = self.pipeline.tokenizer(
            a_with,
            padding="max_length",
            truncation=True,
            max_length=self.pipeline.model.config.max_position_embeddings,
            return_tensors="pt",
        ).to(self.pipeline.device)
        sep_token_indices = [encoding_with.input_ids[i].tolist().index(self.pipeline.tokenizer.sep_token_id) for i in range(len(a))]
        sep_token_indices = torch.tensor(sep_token_indices, device=self.pipeline.device)
        seq_len = encoding_without.input_ids.shape[1]
        all_subword_labels = torch.zeros_like(encoding_without.input_ids, device=self.pipeline.device).float()
        for j in range(1, seq_len):  # Skip [SEP]
            if torch.all(encoding_without.input_ids[:, j] == self.pipeline.tokenizer.pad_token_id):
                break
            input_ids_with = deepcopy(encoding_with.input_ids)
            input_ids_without = deepcopy(encoding_without.input_ids)
            labels_without = -100 * torch.ones_like(encoding_without.input_ids, device=self.pipeline.device)
            labels_with = -100 * torch.ones_like(encoding_with.input_ids, device=self.pipeline.device)
            for i in range(len(a)):
                if j + sep_token_indices[i] >= seq_len:  # Cannot mask due to truncation
                    continue
                if j not in itertools.chain.from_iterable(subwords_by_words_a[i]):  # Not a token of interest
                    continue
                assert encoding_without.input_ids[i, j] == encoding_with.input_ids[i, j + sep_token_indices[i]]
                input_ids_without[i, j] = self.pipeline.tokenizer.mask_token_id
                input_ids_with[i, j + sep_token_indices[i]] = self.pipeline.tokenizer.mask_token_id
                labels_without[i, j] = encoding_without.input_ids[i, j]
                labels_with[i, j + sep_token_indices[i]] = encoding_with.input_ids[i, j + sep_token_indices[i]]
            output_with: MaskedLMOutput = self.pipeline.model(
                input_ids_with,
                attention_mask=encoding_with.attention_mask,
                return_dict=True,
                **kwargs,
            )
            output_without: MaskedLMOutput = self.pipeline.model(
                input_ids_without,
                attention_mask=encoding_without.attention_mask,
                return_dict=True,
                **kwargs,
            )
            loss_with = torch.zeros(len(a), device=self.pipeline.device)
            loss_without = torch.zeros(len(a), device=self.pipeline.device)
            for i in range(len(a)):
                if labels_without[i, j] == -100:
                    continue
                loss_with[i] = torch.nn.functional.cross_entropy(
                    output_with.logits[i, j + sep_token_indices[i]],
                    labels_with[i, j + sep_token_indices[i]],
                )
                loss_without[i] = torch.nn.functional.cross_entropy(
                    output_without.logits[i, j],
                    labels_without[i, j],
                )
            diffs = 1 - torch.maximum(
                torch.tensor(0., device=self.pipeline.device),
                (loss_without - loss_with) / torch.maximum(loss_without, loss_with),
            )
            diffs = diffs.nan_to_num()
            all_subword_labels[:, j] = diffs
        samples = []
        for i in range(len(a)):
            labels = self._subword_labels_to_word_labels(all_subword_labels[i], subwords_by_words_a[i])
            sample = DifferenceSample(
                tokens_a=tuple(a[i].split()),
                tokens_b=tuple(b[i].split()),
                labels_a=tuple(labels),
                labels_b=None,
            )
            samples.append(sample)
        return samples

    def _get_subwords_by_word(self, sentence: str) -> List[Ngram]:
        """
        :return: For each word in the sentence, the positions of the subwords that make up the word.
        """
        batch_encoding = self.pipeline.tokenizer(
            sentence,
            padding=True,
            truncation=True,
        )
        deletable_tokens: List[List[int]] = []
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
            if is_tail and len(deletable_tokens) > 0:
                deletable_tokens[-1].append(subword_idx)
            else:
                deletable_tokens.append([subword_idx])
        return deletable_tokens

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
