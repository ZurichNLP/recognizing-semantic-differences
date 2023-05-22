Code for the paper: "Towards Unsupervised Recognition of Semantic Differences in Related Documents".

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
* `python -m experiments.scripts.create_languages_figure`
* `python -m experiments.scripts.create_document_length_figure`
* `python -m experiments.scripts.create_permutation_figure`
