"""
Lagrangian mechanics via symbolic differentiation.

This module mirrors the core ideas of MIT scmutils:
  - A Lagrangian is a callable  L(t, q, qdot) -> expression
  - lagrange_equations(L) returns the Euler-Lagrange equations for a
    generalized coordinate treated as an unknown function of time.

All symbolic work is delegated to SymPy.
"""

import sympy as sp
from sympy import Function, Symbol, simplify, cos


# ---------------------------------------------------------------------------
# Euler-Lagrange equations
# ---------------------------------------------------------------------------

def lagrange_equations(L_func, q_func: Function, t: Symbol = None):
    """
    Derive the Euler-Lagrange equation for a single generalized coordinate.

    Parameters
    ----------
    L_func : callable(t, q, qdot) -> sympy expression
        The Lagrangian.
    q_func : sympy.Function
        The generalized coordinate as an *undefined* SymPy function
        (analogous to ``literal-function`` in scmutils).
    t : sympy.Symbol, optional
        The independent variable (defaults to ``Symbol('t')``).

    Returns
    -------
    sympy expression
        The Euler-Lagrange equation set equal to zero:
          d/dt (∂L/∂q̇) − ∂L/∂q = 0
    """
    if t is None:
        t = Symbol("t")

    q = q_func(t)
    qdot = q.diff(t)

    L_expr = L_func(t, q, qdot)

    # ∂L/∂q̇
    dL_dqdot = L_expr.diff(qdot)

    # d/dt (∂L/∂q̇)
    ddt_dL_dqdot = dL_dqdot.diff(t)

    # ∂L/∂q
    dL_dq = L_expr.diff(q)

    # Euler-Lagrange: d/dt(∂L/∂q̇) − ∂L/∂q = 0
    return simplify(ddt_dL_dqdot - dL_dq)


# ---------------------------------------------------------------------------
# Multi-coordinate Euler-Lagrange
# ---------------------------------------------------------------------------

def lagrange_equations_multi(L_func, q_funcs, t: Symbol = None):
    """
    Derive Euler-Lagrange equations for multiple generalized coordinates.

    Parameters
    ----------
    L_func : callable(t, qs, qdots) -> sympy expression
        Lagrangian that takes lists of coordinates and velocities.
    q_funcs : list of sympy.Function
        Generalized coordinates.
    t : sympy.Symbol, optional

    Returns
    -------
    list of sympy expressions (each == 0)
    """
    if t is None:
        t = Symbol("t")

    qs = [qf(t) for qf in q_funcs]
    qdots = [q.diff(t) for q in qs]

    L_expr = L_func(t, qs, qdots)

    eqs = []
    for q, qdot in zip(qs, qdots):
        dL_dqdot = L_expr.diff(qdot)
        ddt_dL_dqdot = dL_dqdot.diff(t)
        dL_dq = L_expr.diff(q)
        eqs.append(simplify(ddt_dL_dqdot - dL_dq))
    return eqs


# ---------------------------------------------------------------------------
# Pendulum Lagrangian  (direct port of the Scheme code)
# ---------------------------------------------------------------------------

def L_pendulum(m, l, g):
    """
    Return a Lagrangian for a simple pendulum.

    Scheme original::

        (define ((L-pendulum m l g) local)
          (let ((theta     (coordinate local))
                (theta-dot (velocity local)))
            (- (* 1/2 m (expt (* l theta-dot) 2))
               (* m g l (- 1 (cos theta))))))

    Parameters
    ----------
    m, l, g : sympy symbols or numbers
        Mass, length, gravitational acceleration.

    Returns
    -------
    callable(t, theta, theta_dot) -> sympy expression
    """
    def L(t, theta, theta_dot):
        T = sp.Rational(1, 2) * m * (l * theta_dot) ** 2
        V = m * g * l * (1 - cos(theta))
        return T - V

    return L


# ---------------------------------------------------------------------------
# Convenience wrapper (OOP style)
# ---------------------------------------------------------------------------

class LagrangianMechanics:
    """High-level helper that bundles a Lagrangian with its coordinates."""

    def __init__(self, L_func, q_funcs, t=None):
        self.L_func = L_func
        self.q_funcs = q_funcs if isinstance(q_funcs, (list, tuple)) else [q_funcs]
        self.t = t or Symbol("t")

    def euler_lagrange(self):
        """Return the list of Euler-Lagrange equations (each == 0)."""
        if len(self.q_funcs) == 1:
            return [lagrange_equations(self.L_func, self.q_funcs[0], self.t)]
        return lagrange_equations_multi(
            self.L_func, self.q_funcs, self.t
        )

    def show(self):
        """Pretty-print the equations."""
        eqs = self.euler_lagrange()
        for i, eq in enumerate(eqs):
            print(f"EL equation {i}: {eq} = 0")
        return eqs
