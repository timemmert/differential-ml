import unittest

import torch
from torch.testing import assert_close

from examples.data_generation.functions import (
    OutputFormat,
    quadratic_one_in_one_out,
    quadratic_two_in_one_out,
    polynomial_trigonometric_two_in_one_out,
    trigonometric_two_in_two_out,
)
from examples.data_generation.random_data import uniform_random_in_range


class TestDataGeneration(unittest.TestCase):
    def test_uniform_random_in_range_produces_in_range(self):
        n_test = 1000
        start = 100
        end = 105
        data = uniform_random_in_range(n_test, start=start, end=end)
        assert all(data > start)
        assert all(data < end)

    def test_quadratic_one_dim(self):
        n_test = 1000
        a = 0.1
        data, y, dydx = quadratic_one_in_one_out(
            a, n_test, output_format=OutputFormat.TORCH
        )

        y_manual = (a * data[:, 0] ** 2).reshape(-1, 1)
        dydx_manual = (a * 2 * data).reshape(-1, 1, 1)

        assert_close(y, y_manual)
        assert_close(dydx, dydx_manual)

    def test_quadratic_two_dim(self):
        n_test = 10
        a = 0.1
        data, y, dydx = quadratic_two_in_one_out(
            a, n_test, output_format=OutputFormat.TORCH
        )

        y_manual = (a * data[:, 0] ** 2 * data[:, 1] ** 2).reshape(-1, 1)
        dydx_manual = torch.stack(
            (
                a * 2 * data[:, 0] * data[:, 1] ** 2,
                a * 2 * data[:, 0] ** 2 * data[:, 1],
            ),
            dim=1,
        ).reshape(-1, 1, 2)

        assert_close(y, y_manual)
        assert_close(dydx, dydx_manual)

    def test_poly_trigonometric_two_dim(self):
        # function y = x[0]³*x[1]² + sin(x[0])*x[1]
        # derivative dydx = (3*x[0]²*x[1]² + cos(x[0])*x[1], 2*x[0]³*x[1] + sin(x[0]))
        n_test = 1000

        data, y, dydx = polynomial_trigonometric_two_in_one_out(
            n_test, output_format=OutputFormat.TORCH
        )

        y_manual = (
            data[:, 0] ** 3 * data[:, 1] ** 2 + torch.sin(data[:, 0]) * data[:, 1]
        ).reshape(-1, 1)
        dydx_manual = torch.stack(
            (
                3 * data[:, 0] ** 2 * data[:, 1] ** 2
                + torch.cos(data[:, 0]) * data[:, 1],
                2 * data[:, 0] ** 3 * data[:, 1] + torch.sin(data[:, 0]),
            ),
            dim=1,
        ).reshape(-1, 1, 2)
        assert_close(y, y_manual)
        assert_close(dydx, dydx_manual)

    def test_trigonometric_two_in_two_out(self):
        n_test = 1000
        data, y, dydx = trigonometric_two_in_two_out(
            n_test, output_format=OutputFormat.TORCH
        )

        y_manual = torch.stack((torch.sin(data[:, 0]), torch.cos(data[:, 1])), dim=1)
        dydx_manual = torch.stack(
            (
                torch.stack((torch.cos(data[:, 0]), torch.zeros(n_test)), dim=1),
                torch.stack((torch.zeros(n_test), -torch.sin(data[:, 1])), dim=1),
            ),
            dim=2,
        )
        assert_close(y, y_manual)
        assert_close(dydx, dydx_manual)
