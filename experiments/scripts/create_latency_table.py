import time

from transformers import pipeline

from data.ists import ISTSDataset
from experiments.benchmark import DifferenceRecognitionBenchmark
from recognizers import DiffAlign, DiffDel, DiffMask


device = 0

benchmark = DifferenceRecognitionBenchmark(
    positive_dataset=ISTSDataset(split="test"),
    positive_ratio=1.0,
    num_sentences_per_document=1,
    num_inversions=0,
)

num_tokens = benchmark.num_tokens

recognizers = [
    DiffAlign(
        pipeline=pipeline(
            model="xlm-roberta-base",
            task="feature-extraction",
            device=device,
        ),
        layer=-1,
        batch_size=16,
    ),
    DiffDel(
        pipeline=pipeline(
            model="xlm-roberta-base",
            task="feature-extraction",
            device=device,
        ),
        batch_size=16,
    ),
    DiffMask(
        pipeline=pipeline(
            model="xlm-roberta-base",
            task="fill-mask",
            device=device,
        ),
        batch_size=16,
    ),
]

for recognizer in recognizers:
    print(recognizer)
    start_time = time.time()
    result = benchmark.evaluate(recognizer)
    end_time = time.time()
    elapsed_time_seconds = end_time - start_time
    print(f"Elapsed time: {elapsed_time_seconds:.2f} seconds")
    print(f"Elapsed time per 1k tokens: {1000 * elapsed_time_seconds / num_tokens:.2f} seconds")
