"""
Tests for functionality checks in class SolveDiffusion2D
"""

import pytest
from diffusion2d import SolveDiffusion2D


def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    solver.initialize_domain(w=20., h=10., dx=0.05, dy=0.05)
    solver.initialize_physical_parameters(d=5., T_cold=650., T_hot=750.)
    expected_dt = pytest.approx(0.000125, abs=1e-9)
    assert solver.dt == expected_dt


def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver = SolveDiffusion2D()
    solver.initialize_domain(w=20., h=10., dx=0.05, dy=0.05)
    solver.initialize_physical_parameters(d=5., T_cold=650., T_hot=750.)
    u0 = solver.set_initial_condition()
    expected_u0_center = 750.
    actual_u0_center = u0[solver.nx // 2, solver.ny // 2]
    assert actual_u0_center == expected_u0_center
