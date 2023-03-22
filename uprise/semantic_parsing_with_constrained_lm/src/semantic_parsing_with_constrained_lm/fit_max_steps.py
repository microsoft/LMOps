# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import collections
import itertools
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Quantiles:
    # THe quantile used to find the intercept.
    # In effect, we sort the list of lengths, and take
    # `lengths[int(len(lengths) * intercept_quantile)]`
    # to use as the intercept.
    intercept_quantile: float
    # The quantile used for the slope.
    # For each datum, we compute the slope that would be needed so that the
    # predicted length matches the gold length.
    slope_quantile: float


@dataclass
class Result:
    # Number of test instances that had predicted lengths smaller than gold lengths.
    num_unreachable: int
    # predicted length - gold length
    surplus_steps: List[int]


def fit(pairs: List[Tuple[int, int]]) -> Iterator[Tuple[Quantiles, float, float]]:
    # 0, 0.01, 0.02, ..., 0.99, 1.00
    intercept_quantiles = np.linspace(0, 1, 101)
    # 1, 0.99, 0.98, ..., 0.90
    slope_quantiles = np.linspace(1, 0.9, 11)

    intercept_by_quantile = dict(
        zip(
            intercept_quantiles, np.quantile([o for _, o in pairs], intercept_quantiles)
        )
    )
    intercept_by_quantile[-1] = 0.0

    for intercept_quantile, intercept in intercept_by_quantile.items():
        slopes = [(o - intercept) / i for i, o in pairs]
        max_slopes = np.quantile(slopes, slope_quantiles)
        for slope_quantile, slope in zip(slope_quantiles, max_slopes):
            yield Quantiles(intercept_quantile, slope_quantile), intercept, slope


def cross_validation_fit(
    pairs: List[Tuple[int, int]], num_splits: int
) -> Dict[Quantiles, List[Result]]:
    all_results: Dict[Quantiles, List[Result]] = collections.defaultdict(list)

    # Perform n-fold cross-validation
    test_set_size = len(pairs) / num_splits  # This is a real number
    random.Random(0).shuffle(pairs)
    for test_start, test_end in zip(
        np.arange(0, len(pairs), test_set_size),
        np.arange(test_set_size, len(pairs), test_set_size),
    ):
        # Round down first so that we get an integer size for the test set
        test_start = int(test_start)
        test_end = int(test_end)

        train_data = pairs[:test_start] + pairs[test_end:]
        test_data = pairs[test_start:test_end]

        for quantiles, intercept, slope in fit(train_data):
            predicted = [int(i * slope + intercept) for i, _ in test_data]
            num_missed = sum(p < o for p, (_, o) in zip(predicted, test_data))
            surplus_steps = [p - o for p, (_, o) in zip(predicted, test_data)]
            all_results[quantiles].append(Result(num_missed, surplus_steps))

    return all_results


def filter_fit(
    all_results: Dict[Quantiles, List[Result]], max_unreachable: int
) -> List[Tuple[Quantiles, int, float]]:
    filtered_results: List[Tuple[Quantiles, int, float]] = []
    for q, results in all_results.items():
        total_unreachable = sum(r.num_unreachable for r in results)
        if total_unreachable <= max_unreachable:
            mean_surplus_steps = np.average(
                list(itertools.chain.from_iterable(r.surplus_steps for r in results))
            )
            filtered_results.append((q, total_unreachable, mean_surplus_steps))
    filtered_results.sort(key=lambda x: x[2])

    return filtered_results


def compute_and_print_fit(
    pairs: List[Tuple[int, int]], num_splits: int, max_unreachable: int
) -> Tuple[float, float]:
    all_results: Dict[Quantiles, List[Result]] = cross_validation_fit(pairs, num_splits)

    filtered_results: List[Tuple[Quantiles, int, float]] = filter_fit(
        all_results, max_unreachable
    )

    print(f"Best params for total unreachable < {max_unreachable}:")
    for q, total_unreachable, mean_surplus_steps in filtered_results[:5]:
        print(
            f"Intercept quantile {q.intercept_quantile:.2f}, "
            f"slope quantile {q.slope_quantile:.2f}: "
            f"num unreachable = {total_unreachable}, "
            f"mean surplus steps = {mean_surplus_steps:.3g}"
        )

    min_surplus_steps = filtered_results[0][2]
    all_params_with_min_surplus_steps = [
        q
        for q, _, mean_surplus_steps in filtered_results
        if mean_surplus_steps == min_surplus_steps
    ]
    best_params = min(
        all_params_with_min_surplus_steps,
        key=lambda q: (q.intercept_quantile, q.slope_quantile),
    )

    if best_params.intercept_quantile == -1:
        intercept = 0
    else:
        intercept = np.quantile([o for _, o in pairs], best_params.intercept_quantile)
    slope = np.quantile(
        [(o - intercept) / i for i, o in pairs], best_params.slope_quantile
    )

    print()
    print(f"Final results: intercept = {intercept}, slope = {slope}")
    return (intercept, slope)
