from collections import OrderedDict

from transformers import pipeline

from experiments.benchmark import DifferenceRecognitionResult
from experiments.utils import load_summary_benchmarks
from recognizers import DiffAlign


benchmarks = load_summary_benchmarks("test")
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

simcse_spearmans = [[] for _ in range(len(benchmarks) - 5)]
for recognizer_name, recognizer_results in list(results.items()):
    if "simcse" in recognizer_name.lower():
        for i, result in enumerate(recognizer_results):
            simcse_spearmans[i].append(result.spearman)
        del results[recognizer_name]
assert len(simcse_spearmans[0]) == num_seeds

template = """\
Approach                                                        & iSTS & +\\,Negatives & +\\,Documents & +\\,Permuted & +\\,Cross-lingual \\\\ \\midrule
\\diffalign{}                                                    &      &               &               &           &              \\\\
-- \\xlmr{} + SimCSE                                             & 1.00 & 1.00          & 1.00          & 1.00      & 1.00         \\\\
"""

for spearmans in simcse_spearmans:
    mean = sum(spearmans) / len(spearmans)
    template = template.replace("1.00", f"{100*mean:.1f}", 1)
print(template)
