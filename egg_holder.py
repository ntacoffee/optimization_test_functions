#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Union

import numpy as np
from overrides import overrides

from test_function import TestFunction


class Eggholder(TestFunction):
    def __init__(self) -> None:
        super().__init__(dim=2)

    @overrides
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        first_term = -(x[:, 1] + 47) * np.sin(
            np.sqrt(np.abs(x[:, 0] / 2 + (x[:, 1] + 47)))
        )
        second_term = -x[:, 0] * np.sin(np.sqrt(np.abs(x[:, 0] - (x[:, 1] + 47))))
        return first_term + second_term


if __name__ == "__main__":
    egg_holder = Eggholder()

    value = egg_holder(np.array([512, 404.2319]))
    assert value > -959.6407 - 0.01
    assert value < -959.6407 + 0.01
