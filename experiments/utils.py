from typing import List

from data.ists import ISTSDataset
from data.pawsx import PAWSXDataset, CrosslingualPAWSXDataset
from experiments.benchmark import DifferenceRecognitionBenchmark


def load_summary_benchmarks(split: str) -> List[DifferenceRecognitionBenchmark]:
    """
    Returns a list of benchmark configurations that corresponds to the columns of Tables 1 and 2
    """
    print(f"Loading {split} summary benchmarks...")
    ists_dataset = ISTSDataset(split=("train" if split == "validation" else split))
    pawsx_dataset = PAWSXDataset(split=split)

    benchmarks = []

    # iSTS
    benchmarks.append(DifferenceRecognitionBenchmark(
        positive_dataset=ists_dataset,
        negative_dataset=pawsx_dataset,
        positive_ratio=1.0,
        num_sentences_per_document=1,
        num_inversions=0,
    ))
    # + Negatives
    benchmarks.append(DifferenceRecognitionBenchmark(
        positive_dataset=ists_dataset,
        negative_dataset=pawsx_dataset,
        positive_ratio=0.5,
        num_sentences_per_document=1,
        num_inversions=0,
    ))
    # + Documents
    benchmarks.append(DifferenceRecognitionBenchmark(
        positive_dataset=ists_dataset,
        negative_dataset=pawsx_dataset,
        positive_ratio=0.5,
        num_sentences_per_document=5,
        num_inversions=0,
    ))
    # + Permutations
    benchmarks.append(DifferenceRecognitionBenchmark(
        positive_dataset=ists_dataset,
        negative_dataset=pawsx_dataset,
        positive_ratio=0.5,
        num_sentences_per_document=5,
        num_inversions=5,
    ))

    for lang in ["de", "es", "fr", "ja", "ko", "zh"]:
        benchmarks.append(DifferenceRecognitionBenchmark(
            positive_dataset=ISTSDataset(
                split=("train" if split == "validation" else split),
                tgt_lang=lang,
            ),
            negative_dataset=CrosslingualPAWSXDataset(
                split=split,
                tgt_lang=lang,
            ),
            positive_ratio=0.5,
            num_sentences_per_document=5,
            num_inversions=5,
        ))

    return benchmarks


def load_negative_ratio_benchmarks(split: str) -> List[DifferenceRecognitionBenchmark]:
    """
    Returns a list of benchmark configurations that corresponds to the x-axis of Figure 2a.
    """
    print(f"Loading {split} negative ratio benchmarks...")
    ists_dataset = ISTSDataset(split=("train" if split == "validation" else split))
    pawsx_dataset = PAWSXDataset(split=split)
    benchmarks = []
    for negative_ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        benchmarks.append(DifferenceRecognitionBenchmark(
            positive_dataset=ists_dataset,
            negative_dataset=pawsx_dataset,
            positive_ratio=1-negative_ratio,
            num_sentences_per_document=1,
            num_inversions=0,
        ))
    return benchmarks


def load_document_length_benchmarks(split: str) -> List[DifferenceRecognitionBenchmark]:
    """
    Returns a list of benchmark configurations that corresponds to the x-axis of Figure 2b.
    """
    print(f"Loading {split} document length benchmarks...")
    ists_dataset = ISTSDataset(split=("train" if split == "validation" else split))
    pawsx_dataset = PAWSXDataset(split=split)
    benchmarks = []
    for num_sentences_per_document in range(1, 17):
        benchmarks.append(DifferenceRecognitionBenchmark(
            positive_dataset=ists_dataset,
            negative_dataset=pawsx_dataset,
            positive_ratio=0.5,
            num_sentences_per_document=num_sentences_per_document,
            num_inversions=0,
        ))
    return benchmarks


def load_permutation_benchmarks(split: str) -> List[DifferenceRecognitionBenchmark]:
    """
    Returns a list of benchmark configurations that corresponds to the x-axis of Figure 2c.
    """
    print(f"Loading {split} permutation benchmarks...")
    ists_dataset = ISTSDataset(split=("train" if split == "validation" else split))
    pawsx_dataset = PAWSXDataset(split=split)
    benchmarks = []
    for num_inversions in range(0, 11):
        benchmarks.append(DifferenceRecognitionBenchmark(
            positive_dataset=ists_dataset,
            negative_dataset=pawsx_dataset,
            positive_ratio=0.5,
            num_sentences_per_document=5,
            num_inversions=num_inversions,
        ))
    return benchmarks


def load_language_pairs_benchmarks(split: str) -> List[DifferenceRecognitionBenchmark]:
    """
    Returns a list of benchmark configurations that corresponds to the x-axis of Figure 2d.
    """
    print(f"Loading {split} language pairs benchmarks...")
    benchmarks = []
    for lang in ["en", "de", "es", "fr", "ja", "ko", "zh"]:
        benchmarks.append(DifferenceRecognitionBenchmark(
            positive_dataset=ISTSDataset(
                split=("train" if split == "validation" else split),
                tgt_lang=lang,
            ),
            negative_dataset=CrosslingualPAWSXDataset(
                split=split,
                tgt_lang=lang,
            ),
            positive_ratio=0.5,
            num_sentences_per_document=1,
            num_inversions=0,
        ))
    return benchmarks
