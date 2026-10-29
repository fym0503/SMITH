"""Numerical and runner regression checks for the manuscript's FW/prior settings."""

import importlib.util
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize
import torch

from smith.min_norm_solvers import MinNormSolver
from smith.model_selector import get_model


ROOT = Path(__file__).resolve().parents[1]


def load_runner(relative_path):
    spec = importlib.util.spec_from_file_location('checked_runner', ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SolverChecks(unittest.TestCase):
    def solve(self, gradients):
        array = np.asarray(gradients, dtype=float)
        weights, cost = MinNormSolver.find_min_norm_element_FW(
            [[torch.tensor(row)] for row in array]
        )
        self.assertTrue(np.isfinite(weights).all())
        self.assertGreaterEqual(weights.min(), 0)
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertAlmostEqual(cost, float(np.linalg.norm(weights @ array) ** 2))
        return weights, cost

    def test_two_tasks_against_independent_constrained_optimizer(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            array = rng.normal(size=(2, 8))
            gram = array @ array.T
            weights, cost = self.solve(array)
            oracle = minimize(
                lambda x: x @ gram @ x, [0.5, 0.5],
                jac=lambda x: 2 * gram @ x, method='SLSQP',
                bounds=[(0, 1)] * 2,
                constraints={'type': 'eq', 'fun': lambda x: x.sum() - 1},
                options={'ftol': 1e-12},
            )
            self.assertTrue(oracle.success)
            self.assertAlmostEqual(cost, oracle.fun, places=8)

    def test_three_conflicting_tasks(self):
        weights, cost = self.solve([[2, 0], [0, 1], [-1, -1]])
        np.testing.assert_allclose(weights, [0.2, 0.4, 0.4], atol=1e-4)
        self.assertLess(cost, 1e-8)

    def test_exact_boundary_optimum(self):
        weights, cost = self.solve([[1, 0], [2, 0], [3, 0]])
        np.testing.assert_array_equal(weights, [1, 0, 0])
        self.assertEqual(cost, 1)

    def test_zero_identical_and_single_task(self):
        for gradients in ([[0, 0]] * 3, [[2, -1]] * 3, [[3, 4]]):
            with self.subTest(gradients=gradients):
                self.solve(gradients)

    def test_iteration_cap_and_uniform_initialization(self):
        gradients = [[2, 0], [0, 1], [-1, -1]]
        with patch.object(MinNormSolver, 'MAX_ITER', 0):
            initial, initial_cost = self.solve(gradients)
        np.testing.assert_array_equal(initial, np.full(3, 1 / 3))
        with patch.object(MinNormSolver, 'MAX_ITER', 1):
            weights, cost = self.solve(gradients)
        self.assertLessEqual(cost, initial_cost)
        self.assertFalse(np.allclose(initial, weights))

    def test_multiblock_gradients(self):
        weights, cost = MinNormSolver.find_min_norm_element_FW([
            [torch.tensor([2.0]), torch.tensor([0.0])],
            [torch.tensor([0.0]), torch.tensor([1.0])],
        ])
        np.testing.assert_allclose(weights, [0.2, 0.8])
        self.assertAlmostEqual(cost, 0.8)

    def test_no_tasks(self):
        with self.assertRaises(ValueError):
            MinNormSolver.find_min_norm_element_FW([])


class RunnerChecks(unittest.TestCase):
    def test_both_runners_call_fw_and_update_parameters(self):
        for path in ('scripts/main.py', 'src/smith_agent/resources/scripts/main.py'):
            with self.subTest(runner=path):
                runner = load_runner(path)
                torch.manual_seed(1)
                args = SimpleNamespace(device='cpu', lam=0.5, sigma=0.5, dim=4,
                                       dropout_rate=0, activation='tanh', hurdle=False)
                inputs = torch.randn(8, 4)
                target_a, target_b = torch.randn(8, 2), torch.randn(8, 1)
                tasks = ['input', 'recon', 'coo']
                models = get_model(tasks, [inputs, target_a, target_b], args)
                parameters = [p for m in models.values() for p in m.parameters()]
                optimizer = torch.optim.Adam(parameters, lr=0.001)
                before = models['rep'].mu.detach().clone()
                kwargs = {'device': torch.device('cpu')} if 'device' in inspect.signature(runner.process_batch).parameters else {}
                with patch.object(MinNormSolver, 'find_min_norm_element', side_effect=AssertionError('PGD must not be called')):
                    with patch.object(MinNormSolver, 'find_min_norm_element_FW', wraps=MinNormSolver.find_min_norm_element_FW) as fw:
                        losses = runner.process_batch(
                            models, {'input': inputs, 'recon': target_a, 'coo': target_b},
                            tasks, {'recon': torch.nn.MSELoss(), 'coo': torch.nn.MSELoss()},
                            optimizer, None, {}, **kwargs,
                        )
                        self.assertEqual(fw.call_count, 1)
                self.assertTrue(all(np.isfinite(value) for value in losses.values()))
                self.assertFalse(torch.equal(before, models['rep'].mu.detach()))

    def test_both_runners_initialize_only_priors_to_10000(self):
        for path in ('scripts/main.py', 'src/smith_agent/resources/scripts/main.py'):
            with self.subTest(runner=path), tempfile.TemporaryDirectory() as tmp:
                runner = load_runner(path)
                rep = torch.nn.Module()
                rep.mu = torch.nn.Parameter(torch.zeros(4))
                args = SimpleNamespace(saving_dir=tmp, log_dir=tmp, prior_file='prior.csv', epoch=0, device='cpu')
                with patch.object(runner, 'get_dataset', return_value=(np.array(['A','B','C','D']), ['input'], [np.zeros((2,4))])), \
                     patch.object(runner, 'split_dataset', return_value=([], [])), \
                     patch.object(runner.losses, 'get_loss', return_value={}), \
                     patch.object(runner.model_selector, 'get_model', return_value={'rep': rep}), \
                     patch.object(runner.model_selector, 'get_optimizer', return_value=None), \
                     patch.object(runner, 'get_prior_idx', return_value=[1, 3]):
                    runner.Smith(args)
                torch.testing.assert_close(rep.mu.detach(), torch.tensor([0.0, 10000.0, 0.0, 10000.0]))


if __name__ == '__main__':
    unittest.main()
