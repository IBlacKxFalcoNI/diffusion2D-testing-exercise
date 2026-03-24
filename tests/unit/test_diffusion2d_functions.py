"""
Tests for functions in class SolveDiffusion2D
"""

import pytest

from diffusion2d import SolveDiffusion2D


def test_initialize_domain():
    """
    Check function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    solver.initialize_domain(w=20., h=10., dx=0.05, dy=0.05)
    expected_w = 20.
    actual_w = solver.w
    assert actual_w == expected_w
    assert solver.h == 10.
    assert solver.dx == 0.05
    assert solver.dy == 0.05
    assert solver.nx == 400
    assert solver.ny == 200


def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_physical_parameters
    """
    solver = SolveDiffusion2D()
    solver.dx = 0.1
    solver.dy = 0.1
    solver.initialize_physical_parameters(d=5., T_cold=650., T_hot=750.)
    assert solver.D == 5.
    assert solver.T_cold == 650.
    assert solver.T_hot == 750.
    expected_dt = pytest.approx(0.0005, abs=1e-6)
    assert solver.dt == expected_dt

def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.set_initial_condition
    """
    solver = SolveDiffusion2D()
    solver.w = 10.
    solver.h = 10.
    solver.dx = 0.1
    solver.dy = 0.1
    solver.T_cold = 650.
    solver.T_hot = 750.
    solver.nx = int(solver.w / solver.dx)
    solver.ny = int(solver.h / solver.dy)

    u0 = solver.set_initial_condition()

    assert u0.shape == (solver.nx, solver.ny)
