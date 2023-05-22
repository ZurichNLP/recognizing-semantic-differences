import numpy as np
from transformers import pipeline

from experiments.benchmark import DifferenceRecognitionResult
from experiments.utils import load_summary_benchmarks
from recognizers.diff_del import DiffDel, DiffDelWithReencode


benchmarks = load_summary_benchmarks("validation")
device = 0
num_seeds = 10

seeds = [22417, 26186, 28852, 39168, 43002, 73246, 75213, 75370, 92253, 96301]

recognizers = []
# standard
for seed in seeds[:num_seeds]:
    recognizers.append(DiffDel(
        pipeline=pipeline(
            model=f"../SimCSE/models/simcse-xlm-roberta-base-mean-pooling-seed{seed}",
            task="feature-extraction",
        ),
        batch_size=4,
        min_n=1,
        max_n=1,
    ))

# -- unigrams and bigrams
for seed in seeds[:num_seeds]:
    recognizers.append(DiffDel(
        pipeline=pipeline(
            model=f"../SimCSE/models/simcse-xlm-roberta-base-mean-pooling-seed{seed}",
            task="feature-extraction",
        ),
        batch_size=4,
        min_n=1,
        max_n=2,
    ))

# -- unigrams, bigrams and trigrams
for seed in seeds[:num_seeds]:
    recognizers.append(DiffDel(
        pipeline=pipeline(
            model=f"../SimCSE/models/simcse-xlm-roberta-base-mean-pooling-seed{seed}",
            task="feature-extraction",
        ),
        batch_size=4,
        min_n=1,
        max_n=3,
    ))

# -- unigrams with re-encoding
for seed in seeds[:num_seeds]:
    recognizers.append(DiffDelWithReencode(
        pipeline=pipeline(
            model=f"../SimCSE/models/simcse-xlm-roberta-base-mean-pooling-seed{seed}",
            task="feature-extraction",
        ),
        batch_size=4,
    ))


results = []
for i, recognizer in enumerate(recognizers):
    print(recognizer)
    recognizer.pipeline.device = device
    recognizer.device = device
    recognizer.pipeline.model = recognizer.pipeline.model.to(device)
    recognizer_results = []
    for benchmark in benchmarks:
        print(benchmark)
        result = benchmark.evaluate(recognizer)
        print(result)
        recognizer_results.append(result)

    # Average last six results (cross-lingual)
    recognizer_results, cross_lingual_results = recognizer_results[:-6], recognizer_results[-6:]
    cross_lingual_mean = sum([result.spearman for result in cross_lingual_results]) / len(cross_lingual_results)
    recognizer_results.append(DifferenceRecognitionResult(spearman=cross_lingual_mean))
    results.append(recognizer_results)
    recognizers[i] = None
    del recognizer

num_rows = 4
num_benchmarks = len(benchmarks) - 5
assert len(results) == num_rows * num_seeds

all_results = np.zeros((num_rows * num_seeds, num_benchmarks))
for i, recognizer_results in enumerate(results):
    for j, result in enumerate(recognizer_results):
        all_results[i, j] = result.spearman

all_results = all_results.reshape((num_rows, num_seeds, num_benchmarks))
mean_results = np.mean(all_results, axis=1)

template = """\
Approach                                                        & iSTS & +\\,Negatives & +\\,Documents & +\\,Permuted & +\\,Cross-lingual \\\\ \\midrule
\\diffdel{} \\xlmr{} + SimCSE      &  0.00 & 0.00          & 0.00          & 0.00      & 0.00        \\\\
-- unigrams and bigrams    &  0.00 & 0.00          & 0.00          & 0.00      & 0.00        \\\\
-- unigrams, bigrams and trigrams    &  0.00 & 0.00          & 0.00          & 0.00      & 0.00        \\\\
-- unigrams with re-encoding    &  0.00 & 0.00          & 0.00          & 0.00      & 0.00        \\\\  \\bottomrule
"""

for result in mean_results.flatten():
    template = template.replace("0.00", f"{100*result:.1f}", 1)

print(template)
