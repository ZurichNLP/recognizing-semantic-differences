from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from tokenizers.pre_tokenizers import Whitespace
from torch import Tensor


@dataclass
class DifferenceSample:
    tokens_a: Tuple[str, ...]
    tokens_b: Tuple[str, ...]
    labels_a: Tuple[float, ...]
    labels_b: Optional[Tuple[float, ...]]


def tokenize(text: str) -> Tuple[str]:
    """
    Apply Moses-like tokenization to a string.
    """
    whitespace_tokenizer = Whitespace()
    output = whitespace_tokenizer.pre_tokenize_str(text)
    # [('This', (0, 4)), ('is', (5, 7)), ('a', (8, 9)), ('test', (10, 14)), ('.', (14, 15))]
    tokens = [str(token[0]) for token in output]
    return tuple(tokens)


def cos_sim(a: Tensor, b: Tensor):
    """
    Copied from https://github.com/UKPLab/sentence-transformers/blob/d928410803bb90f555926d145ee7ad3bd1373a83/sentence_transformers/util.py#L31

    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def pairwise_dot_score(a: Tensor, b: Tensor):
    """
    Copied from https://github.com/UKPLab/sentence-transformers/blob/d928410803bb90f555926d145ee7ad3bd1373a83/sentence_transformers/util.py#L73

    Computes the pairwise dot-product dot_prod(a[i], b[i])
    :return: Vector with res[i] = dot_prod(a[i], b[i])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    return (a * b).sum(dim=-1)


def normalize_embeddings(embeddings: Tensor):
    """
    Copied from https://github.com/UKPLab/sentence-transformers/blob/d928410803bb90f555926d145ee7ad3bd1373a83/sentence_transformers/util.py#L101

    Normalizes the embeddings matrix, so that each sentence embedding has unit length
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def pairwise_cos_sim(a: Tensor, b: Tensor):
    """
    Copied from https://github.com/UKPLab/sentence-transformers/blob/d928410803bb90f555926d145ee7ad3bd1373a83/sentence_transformers/util.py#L87

    Computes the pairwise cossim cos_sim(a[i], b[i])
    :return: Vector with res[i] = cos_sim(a[i], b[i])
   """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    return pairwise_dot_score(normalize_embeddings(a), normalize_embeddings(b))
