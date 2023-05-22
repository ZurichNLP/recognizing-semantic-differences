from collections import OrderedDict

import numpy as np
from transformers import pipeline

from experiments.benchmark import DifferenceRecognitionResult
from experiments.utils import load_language_pairs_benchmarks
from recognizers import DiffAlign


benchmarks = load_language_pairs_benchmarks("validation")
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
        xticklabels={\\textsc{en}, \\textsc{de}, \\textsc{es}, \\textsc{fr}, \\textsc{ja}, \\textsc{ko}, \\textsc{zh}},
    xlabel={Target language},
    ylabel={Spearman correlation},
    enlargelimits=0.05,
    ymin=0, ymax=100,
    legend pos=north west,
    legend cell align={left},
    ybar interval=0.7,
]
\\addplot[
    color=blue,
    fill=blue!40,
    ]
        coordinates {(0,0.1) (1,0.2) (2,0.3) (3,0.4) (4,0.5) (5,0.6) (6,0.7)};

    \\addlegendentry{\\xlmr{} + SimCSE (unsupervised)}
\\addplot[
    color=gray,
    fill=gray!40,
    ]
        coordinates {(0,0.1) (1,0.2) (2,0.3) (3,0.4) (4,0.5) (5,0.6) (6,0.7)};
    \\addlegendentry{\\xlmr{} trained on paraphrases}
\\end{axis}
\\end{tikzpicture}
"""

for recognizer_results in reversed(list(results.values())):
    bars = ""
    for i, result in enumerate(recognizer_results):
        bars += f"({i},{result.spearman * 100:.1f}) "
    bars += "(7, 0)"  # Dummy
    template = template.replace("(0,0.1) (1,0.2) (2,0.3) (3,0.4) (4,0.5) (5,0.6) (6,0.7)", bars, 1)
print(template)
