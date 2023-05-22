from collections import OrderedDict

import numpy as np
from transformers import pipeline

from experiments.benchmark import DifferenceRecognitionResult
from experiments.utils import load_document_length_benchmarks
from recognizers import DiffAlign


benchmarks = load_document_length_benchmarks("validation")
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
        batch_size=4,
    ))
recognizers.append(DiffAlign(
    pipeline=pipeline(
        model="sentence-transformers/paraphrase-xlm-r-multilingual-v1",
        task="feature-extraction",
    ),
    batch_size=4,
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
    xlabel={Number of sentences in document},
    ylabel={Spearman correlation},
    xmin=1, xmax=16,
    ymin=0, ymax=100,
    xtick={2,4,6,8,10,12,14,16},
    ytick={0,20,40,60,80,100},
    legend pos=south west,
    legend cell align={left},
    ymajorgrids=true,
    grid style=dashed,
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

document_lengths = list(range(1, 17))
for recognizer_results in reversed(list(results.values())):
    assert len(recognizer_results) == len(document_lengths)
    line: str = ""
    for document_length, result in zip(document_lengths, recognizer_results):
        line += f"({document_length},{result.spearman*100:.1f})"
    template = template.replace("(0,0.00)(1,0.00)", line, 1)
print(template)
