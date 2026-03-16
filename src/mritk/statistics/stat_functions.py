from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class Statistic:
    name: str
    func: Callable

    def __call__(self, data) -> Any:
        return self.func(data)


Median = Statistic("median", lambda x: np.median(x))
Mean = Statistic("mean", lambda x: np.mean(x))
Std = Statistic("std", lambda x: np.std(x))
Sum = Statistic("sum", lambda x: np.sum(x))
Min = Statistic("min", lambda x: np.min(x))
Max = Statistic("max", lambda x: np.max(x))


@dataclass
class PCx(Statistic):
    percentile: int

    def __init__(self, percentile) -> None:
        super().__init__(f"PC{percentile}", lambda x: np.percentile(x, percentile))
        self.percentile = percentile


# Etc
PC1 = PCx(1)
PC5 = PCx(5)
PC25 = PCx(25)
PC75 = PCx(75)
PC95 = PCx(95)
PC99 = PCx(99)


@dataclass
class StableStatistic(Statistic):
    low: int
    high: int

    def __call__(self, data) -> Any:
        low_value = np.percentile(data, self.low)
        high_value = np.percentile(data, self.high)
        return super().__call__(data[(data > low_value) & (data < high_value)])


StableMean = StableStatistic("stable_mean", lambda x: np.mean(x), 5, 95)
StableStd = StableStatistic("stable_std", lambda x: np.std(x), 5, 95)
