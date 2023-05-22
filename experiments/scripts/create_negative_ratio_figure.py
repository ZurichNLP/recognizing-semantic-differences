from collections import OrderedDict

import numpy as np
from transformers import pipeline

from experiments.benchmark import DifferenceRecognitionResult
from experiments.utils import load_negative_ratio_benchmarks
from recognizers import DiffAlign


benchmarks = load_negative_ratio_benchmarks("validation")
device = 0
num_seeds = 10

seeds = [22417, 26186, 28852, 39168, 43002, 73246, 75213, 75370, 92253, 96301]

recognizers = []
for seed in seeds[:num_seeds]:
    recognizers.append(DiffAlign(
        pipeline=pipeline(
            model=f"../SimCSE/models/simcse-xlm-roberta-base-mean-pooling-seed{seed}",
            task="feature-extraction",
        ),
    ))
recognizers.append(DiffAlign(
    pipeline=pipeline(
        model="sentence-transformers/paraphrase-xlm-r-multilingual-v1",
        task="feature-extraction",
    ),
))

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
    results[str(recognizer)] = recognizer_results
    recognizers[i] = None
    del recognizer

simcse_spearmans = [[] for _ in range(len(benchmarks))]
for recognizer_name, recognizer_results in list(results.items()):
    if "simcse" in recognizer_name.lower():
        for i, result in enumerate(recognizer_results):
            simcse_spearmans[i].append(result.spearman)
        del results[recognizer_name]
results["\\xlmr{} + SimCSE (unsupervised)"] = [DifferenceRecognitionResult(spearman=np.mean(spearmans)) for spearmans in simcse_spearmans]

template = """\
\\begin{tikzpicture}
\\begin{axis}[
    xlabel={Ratio of negative examples},
    ylabel={Spearman correlation},
    xmin=0, xmax=1,
    ymin=0, ymax=100,
    xtick={0.2,0.4,0.6,0.8},
    ytick={0,20,40,60,80,100},
    legend pos=south west,
    legend cell align={left},
    ymajorgrids=true,
    grid style=dashed,
    xticklabel={\\pgfmathparse{\\tick*100}\\pgfmathprintnumber{\\pgfmathresult}\\%},
    point meta={x*100},
]

\\addplot[
    color=blue,
    ]
    coordinates {
    (0,0.00)(1,0.00)
    };
    \\addlegendentry{\\xlmr{} + SimCSE (unsupervised)}
\\addplot[
    color=gray,
    ]
    coordinates {
    (0,0.00)(1,0.00)
    };
    \\addlegendentry{\\xlmr{} trained on paraphrases}
\\end{axis}
\\end{tikzpicture}
"""

negative_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for recognizer_results in reversed(list(results.values())):
    assert len(recognizer_results) == len(negative_ratios)
    line: str = ""
    for negative_ratio, result in zip(negative_ratios, recognizer_results):
        line += f"({negative_ratio},{result.spearman*100:.1f})"
    template = template.replace("(0,0.00)(1,0.00)", line, 1)
print(template)
