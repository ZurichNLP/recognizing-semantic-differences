from transformers import pipeline

from experiments.benchmark import DifferenceRecognitionResult
from experiments.utils import load_summary_benchmarks
from recognizers import DiffMask


benchmarks = load_summary_benchmarks("validation")
device = 0

recognizers = [
    DiffMask(
        pipeline=pipeline(
            model="xlm-roberta-base",
            task="fill-mask",
            device=device,
        ),
        batch_size=4,
    ),
]

results = []
for recognizer in recognizers:
    print(recognizer)
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

template = """\
\\diffmask{}                                                      &      &               &               &           &              \\\\
-- \\xlmr{}                                                      & 0.00 & 0.00          & 0.00          & 0.00      & 0.00         \\\\ \\bottomrule
"""

for recognizer_results in results:
    for result in recognizer_results:
        template = template.replace("0.00", f"{100*result.spearman:.1f}", 1)
print(template)
