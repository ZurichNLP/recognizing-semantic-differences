import numpy as np
from transformers import pipeline

from experiments.utils import load_permutation_benchmarks
from recognizers import DiffAlign, DiffDel, DiffMask

benchmarks = load_permutation_benchmarks("validation")
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
    xlabel={Inversion number},
    ylabel={Spearman correlation},
    xmin=0, xmax=10,
    ymin=0, ymax=100,
    xtick={0,1,2,3,4,5,6,7,8,9,10},
    ytick={0,20,40,60,80,100},
    legend pos=north east,
    legend cell align={left},
    ymajorgrids=true,
    grid style=dashed,
]

\\addplot[
        color=blue,
        style=solid,
    ]
    coordinates {
    (0,0.00)(1,0.00)
    };
    \\addlegendentry{\\diffalign{}}
\\addplot[
        color=red,
        style=dashed,
    ]
    coordinates {
    (0,0.00)(1,0.00)
    };
    \\addlegendentry{\\diffdel{}}
\\addplot[
        color=black,
        style=dotted,
        line width=1.5pt,
    ]
    coordinates {
    (0,0.00)(1,0.00)
    };
    \\addlegendentry{\\diffmask{}}
\\end{axis}
\\end{tikzpicture}
"""

inversion_numbers = list(range(0, 11))
for recognizer_results in [diffalign_results, diffdel_results, diffmask_results]:
    assert len(recognizer_results) == len(inversion_numbers)
    line: str = ""
    for inversion_number, result in zip(inversion_numbers, recognizer_results):
        line += f"({inversion_number},{result * 100:.1f})"
    template = template.replace("(0,0.00)(1,0.00)", line, 1)
print(template)
