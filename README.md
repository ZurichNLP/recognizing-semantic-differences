# Recognizing Semantic Differences (RSD)
![Master](https://github.com/ZurichNLP/recognizing-semantic-differences/workflows/unittest/badge.svg)

Code for the EMNLP 2023 paper ["Towards Unsupervised Recognition of Token-level Semantic Differences in Related Documents"](https://doi.org/10.48550/arXiv.2305.13303).

## Bugfix 2025-06-23
The original version of the code unintentionally selected the non-paraphrases of PAWSX as negative examples, instead of the paraphrases. The bug was fixed in commit https://github.com/ZurichNLP/recognizing-semantic-differences/commit/e94de3cd823389ddb3dd860c85eb169e4a74c843.
Thanks to [@mywitt](https://github.com/miwytt) for pointing out the error.

## Installation

* Requires Python >= 3.7 and PyTorch
* `pip install -r requirements.txt`

## Usage
```python
from recognizers import DiffAlign, DiffDel, DiffMask

diff_align = DiffAlign("ZurichNLP/unsup-simcse-xlm-roberta-base")

a = "Chinese shares close higher Friday ."
b = "Chinese shares close lower Wednesday ."
result = diff_align.predict(a, b)
# DifferenceSample(
#   tokens_a=('Chinese', 'shares', 'close', 'higher', 'Friday', '.'),
#   tokens_b=('Chinese', 'shares', 'close', 'lower', 'Wednesday', '.'),
#   labels_a=(0.07324671745300293, 0.06292498111724854, 0.082577645778656, 0.1421372890472412, 0.2610551714897156, 0.1118348240852356),
#   labels_b=(0.07324671745300293, 0.06292498111724854, 0.082577645778656, 0.1421372890472412, 0.2709317207336426, 0.1118348240852356)
# )
```

## Reproducing the Results
### Table 1: Validation results
* `python -m experiments.scripts.create_validation_table`
### Table 2: Test results
* `python -m experiments.scripts.create_test_table`
### Table 3: Dataset statistics
* `python -m experiments.scripts.create_dataset_statistics_table`
### Table 4: Latency measurements
* `python -m experiments.scripts.create_latency_table`
### Table 5: Ablations for diff<sub>del</sub>
* `python -m experiments.scripts.create_validation_table_del_ablations`
### Figure 2: Additional results
* `python -m experiments.scripts.create_negative_ratio_figure`
* `python -m experiments.scripts.create_document_length_figure`
* `python -m experiments.scripts.create_permutation_figure`
* `python -m experiments.scripts.create_languages_figure`

## Citation
```bibtex
@inproceedings{vamvas-sennrich-2023-rsd,
      title={Towards Unsupervised Recognition of Token-level Semantic Differences in Related Documents},
      author={Jannis Vamvas and Rico Sennrich},
      month = dec,
      year = "2023",
      booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
      address = "Singapore",
      publisher = "Association for Computational Linguistics",
}
```
