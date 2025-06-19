import numpy as np
from transformers import pipeline

from experiments.utils import load_language_pairs_benchmarks
from recognizers import DiffAlign, DiffDel, DiffMask

benchmarks = load_language_pairs_benchmarks("validation")
device = 0

seeds = [22417, 26186, 28852, 39168, 43002, 73246, 75213, 75370, 92253, 96301]
num_seeds = len(seeds)

diffalign_recognizers = [
    DiffAlign(
        pipeline=pipeline(
            model=f"../SimCSE/models/simcse-xlm-roberta-base-mean-pooling-seed{seed}",
            task="feature-extraction",
        ),
    )
    for seed in seeds[:num_seeds]
]
diffdel_recognizers = [
    DiffDel(
        pipeline=pipeline(
            model=f"../SimCSE/models/simcse-xlm-roberta-base-mean-pooling-seed{seed}",
            task="feature-extraction",
        ),
    )
    for seed in seeds[:num_seeds]
]
diffmask_recognizers = [
    DiffMask(
        pipeline=pipeline(
            model="xlm-roberta-base",
            task="fill-mask",
        ),
    )
]

diffalign_results = np.zeros((len(diffalign_recognizers), len(benchmarks)))
for i, recognizer in enumerate(diffalign_recognizers):
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
    diffalign_results[i, :] = [result.spearman for result in recognizer_results]
    diffalign_recognizers[i] = None
    del recognizer
diffalign_results = np.mean(diffalign_results, axis=0)
assert len(diffalign_results) == len(benchmarks)

diffdel_results = np.zeros((len(diffdel_recognizers), len(benchmarks)))
for i, recognizer in enumerate(diffdel_recognizers):
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
    diffdel_results[i, :] = [result.spearman for result in recognizer_results]
    diffdel_recognizers[i] = None
    del recognizer
diffdel_results = np.mean(diffdel_results, axis=0)
assert len(diffdel_results) == len(benchmarks)

diffmask_results = np.zeros((len(diffmask_recognizers), len(benchmarks)))
for i, recognizer in enumerate(diffmask_recognizers):
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
    diffmask_results[i, :] = [result.spearman for result in recognizer_results]
    diffmask_recognizers[i] = None
    del recognizer
diffmask_results = np.mean(diffmask_results, axis=0)
assert len(diffmask_results) == len(benchmarks)

template = """\
\\begin{tikzpicture}
\\begin{axis}[
        xticklabels={\\textsc{en}, \\textsc{de}, \\textsc{es}, \\textsc{fr}, \\textsc{ja}, \\textsc{ko}, \\textsc{zh}},
    xlabel={Target language},
    ylabel={Spearman correlation},
    enlargelimits=0.05,
    ymin=0, ymax=100,
    legend pos=north east,
    legend cell align={left},
    ybar interval=0.7,
]
\\addplot[
    color=blue,
    fill=blue!40,
    ]
        coordinates {(0,0.1) (1,0.2) (2,0.3) (3,0.4) (4,0.5) (5,0.6) (6,0.7)};

    \\addlegendentry{\\footnotesize{\\diffalign{}}}
\\addplot[
    color=red,
    fill=red!40,
    ]
        coordinates {(0,0.1) (1,0.2) (2,0.3) (3,0.4) (4,0.5) (5,0.6) (6,0.7)};
    \\addlegendentry{\\footnotesize{\\diffdel{}}}
\\addplot[
    color=black,
    fill=black!40,
    ]
        coordinates {(0,0.1) (1,0.2) (2,0.3) (3,0.4) (4,0.5) (5,0.6) (6,0.7)};
    \\addlegendentry{\\footnotesize{\\diffmask{}}}
\\end{axis}
\\end{tikzpicture}
"""

for recognizer_results in [diffalign_results, diffdel_results, diffmask_results]:
    bars = ""
    for i, result in enumerate(recognizer_results):
        bars += f"({i},{result * 100:.1f}) "
    bars += "(7, 0)"  # Dummy
    template = template.replace("(0,0.1) (1,0.2) (2,0.3) (3,0.4) (4,0.5) (5,0.6) (6,0.7)", bars, 1)
print(template)
