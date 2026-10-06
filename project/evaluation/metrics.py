# Copyright 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# normalize_answer adapted from Search-R1/verl/utils/reward_score/qa_em.py.
"""Canonical-answer EM/token F1 with 2Wiki/Hotpot special yes/no handling.

Reference: Alab-NII/2wikimultihop/2wikimultihop_evaluate.py.
Entity-alias expansion is deliberately not claimed without id_aliases.json.
"""
from collections import Counter
import re
import string


def normalize_answer(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def score_answer(prediction, reference):
    predicted, expected = normalize_answer(prediction), normalize_answer(reference)
    em = float(predicted == expected)
    if predicted != expected and ({predicted, expected} & {"yes", "no", "noanswer"}):
        return {"em": em, "f1": 0.0}
    left, right = predicted.split(), expected.split()
    common = sum((Counter(left) & Counter(right)).values())
    f1 = 2 * common / (len(left) + len(right)) if common else 0.0
    return {"em": em, "f1": f1}
