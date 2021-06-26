#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from abc import abstractmethod
from typing import Union

import numpy as np


class TestFunction:
    def __init__(self, dim: int) -> None:
        self._dim = dim

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = self._arrange_dim(x)
        return self._evaluate(x)

    @property
    def dim(self):
        return self._dim

    def _arrange_dim(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            if x.shape[0] != self._dim:
                raise Exception("Dimension does not match")
            return x[np.newaxis, :]

        elif x.ndim == 2:
            if x.shape[1] != self._dim:
                raise Exception("Dimension does not match")
            return x

        else:
            raise Exception("Illegal dimension")

    @abstractmethod
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        pass


if __name__ == "__main__":
    pass
