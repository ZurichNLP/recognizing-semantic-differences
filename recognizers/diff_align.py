from typing import List

import torch

from recognizers.feature_based import FeatureExtractionRecognizer
from recognizers.utils import DifferenceSample, cos_sim


class DiffAlign(FeatureExtractionRecognizer):

    def __str__(self):
        return f"DiffAlign(model={self.pipeline.model.name_or_path}, layer={self.layer}"

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
        subword_labels_a = []
        subword_labels_b = []
        for i in range(len(a)):
            cosine_similarities = cos_sim(outputs_a[i], outputs_b[i])
            max_similarities_a = torch.max(cosine_similarities, dim=1).values
            max_similarities_b = torch.max(cosine_similarities, dim=0).values
            subword_labels_a.append((1 - max_similarities_a))
            subword_labels_b.append((1 - max_similarities_b))
        samples = []
        for i in range(len(a)):
            labels_a = self._subword_labels_to_word_labels(subword_labels_a[i], subwords_by_words_a[i])
            labels_b = self._subword_labels_to_word_labels(subword_labels_b[i], subwords_by_words_b[i])
            samples.append(DifferenceSample(
                tokens_a=tuple(a[i].split()),
                tokens_b=tuple(b[i].split()),
                labels_a=tuple(labels_a),
                labels_b=tuple(labels_b),
            ))
        return samples
