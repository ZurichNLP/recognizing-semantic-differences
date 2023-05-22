from collections import OrderedDict

import numpy as np
from transformers import pipeline

from experiments.benchmark import DifferenceRecognitionResult
from experiments.utils import load_summary_benchmarks
from recognizers import DiffAlign, DiffDel, DiffMask

benchmarks = load_summary_benchmarks("validation")
device = 0
num_seeds = 10

seeds = [22417, 26186, 28852, 39168, 43002, 73246, 75213, 75370, 92253, 96301]

recognizers = [
    DiffAlign(
        pipeline=pipeline(
            model="xlm-roberta-base",
            task="feature-extraction",
        ),
        layer=-1,
        batch_size=4,
    ),
    DiffAlign(
        pipeline=pipeline(
            model="xlm-roberta-base",
            task="feature-extraction",
        ),
        layer=8,
        batch_size=4,
    ),
]

for seed in seeds[:num_seeds]:
    recognizers.append(DiffAlign(
        pipeline=pipeline(
            model=f"../SimCSE/models/simcse-xlm-roberta-base-mean-pooling-seed{seed}",
            task="feature-extraction",
        ),
        batch_size=4,
    ))

recognizers.append(
    DiffAlign(
        pipeline=pipeline(
            model="sentence-transformers/paraphrase-xlm-r-multilingual-v1",
            task="feature-extraction",
        ),
        batch_size=4,
    )
)

for seed in seeds[:num_seeds]:
    recognizers.append(DiffDel(
        pipeline=pipeline(
            model=f"../SimCSE/models/simcse-xlm-roberta-base-mean-pooling-seed{seed}",
            task="feature-extraction",
        ),
        batch_size=4,
    ))

recognizers.append(
    DiffDel(
        pipeline=pipeline(
            model="sentence-transformers/paraphrase-xlm-r-multilingual-v1",
            task="feature-extraction",
        ),
        batch_size=4,
    ),
)
recognizers.append(
    DiffMask(
        pipeline=pipeline(
            model="xlm-roberta-base",
            task="fill-mask",
        ),
        batch_size=4,
    ),
)

results = OrderedDict()
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
    results[str(recognizer)] = recognizer_results
    recognizers[i] = None
    del recognizer

simcse_spearmans_diffalign = [[] for _ in range(len(benchmarks) - 5)]
simcse_spearmans_diffdel = [[] for _ in range(len(benchmarks) - 5)]
for recognizer_name, recognizer_results in list(results.items()):
    if "simcse" in recognizer_name.lower():
        if "DiffAlign" in recognizer_name:
            for i, result in enumerate(recognizer_results):
                simcse_spearmans_diffalign[i].append(result.spearman)
        elif "DiffDel" in recognizer_name:
            for i, result in enumerate(recognizer_results):
                simcse_spearmans_diffdel[i].append(result.spearman)
        del results[recognizer_name]
assert len(simcse_spearmans_diffalign) == len(simcse_spearmans_diffdel)
assert len(simcse_spearmans_diffalign[0]) == num_seeds

template = """\
Approach                                                        & iSTS & +\\,Negatives & +\\,Documents & +\\,Permuted & +\\,Cross-lingual \\\\ \\midrule
\\diffalign{}                                                    &      &               &               &           &              \\\\
-- \\xlmr{} (last layer)                                         & 0.00 & 0.00          & 0.00          & 0.00      & 0.00         \\\\
-- \\xlmr{} (8th layer)                                          & 0.00 & 0.00          & 0.00          & 0.00      & 0.00         \\\\
-- \\xlmr{} + SimCSE                                             & 1.00 & 1.00          & 1.00          & 1.00      & 1.00         \\\\
-- \\textit{\\xlmr{} trained on paraphrases}  & \\textit{0.00} & \\textit{0.00} & \\textit{0.00} & \\textit{0.00} & \\textit{0.00}         \\\\  \\midrule
\\diffdel{}                                                    &      &               &               &           &              \\\\
-- \\xlmr{} + SimCSE                                             & 1.00 & 1.00          & 1.00          & 1.00      & 1.00         \\\\
-- \\textit{\\xlmr{} trained on paraphrases}  & \\textit{0.00} & \\textit{0.00} & \\textit{0.00} & \\textit{0.00} & \\textit{0.00}         \\\\  \\midrule
\\diffmask{}                                                      &      &               &               &           &              \\\\
-- \\xlmr{}                                                      & 0.00 & 0.00          & 0.00          & 0.00      & 0.00         \\\\ \\bottomrule
"""

for recognizer_name, recognizer_results in results.items():
    for result in recognizer_results:
        template = template.replace("0.00", f"{100*result.spearman:.1f}", 1)
for spearmans in simcse_spearmans_diffalign + simcse_spearmans_diffdel:
    mean = np.mean(spearmans)
    stddev = np.std(spearmans)
    template = template.replace("1.00", f"{100*mean:.1f}", 1)
print(template)
