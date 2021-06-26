#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import plotly.graph_objects as go
from overrides import overrides

from test_function import TestFunction


class DropWave(TestFunction):
    def __init__(self) -> None:
        super().__init__(dim=2)

    @overrides
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        denominator = 1 + np.cos(12 * np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2))
        numerator = 0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2) + 2
        return -denominator / numerator


if __name__ == "__main__":
    drop_wave = DropWave()

    value = drop_wave(np.array([0, 0]))
    assert value > -1 - 0.01
    assert value < -1 + 0.01

    value = drop_wave(np.array([[0, 0], [0, 0]]))
    assert value[0] > -1 - 0.01
    assert value[0] < -1 + 0.01
    assert value[1] > -1 - 0.01
    assert value[1] < -1 + 0.01

    n_grid = 500
    x = np.linspace(-5.12, 5.12, n_grid)
    y = np.linspace(-5.12, 5.12, n_grid)
    xx, yy = np.meshgrid(x, y)
    z = drop_wave(
        np.concatenate([np.reshape(xx, (-1, 1)), np.reshape(yy, (-1, 1))], axis=1)
    ).reshape((n_grid, n_grid))

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(
        scene=go.layout.Scene(
            aspectmode="manual",
            aspectratio=go.layout.scene.Aspectratio(x=1, y=1, z=0.5),
        )
    )
    fig.show()
