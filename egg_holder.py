#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Union

import numpy as np
import plotly.graph_objects as go
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

    value = egg_holder(np.array([[512, 404.2319], [512, 404.2319]]))
    assert value[0] > -959.6407 - 0.01
    assert value[0] < -959.6407 + 0.01
    assert value[1] > -959.6407 - 0.01
    assert value[1] < -959.6407 + 0.01

    n_grid = 500
    x = np.linspace(-500, 500, n_grid)
    y = np.linspace(-500, 500, n_grid)
    xx, yy = np.meshgrid(x, y)
    z = egg_holder(
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
