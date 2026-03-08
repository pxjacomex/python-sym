"""
Lagrangian mechanics via symbolic differentiation.

This module mirrors the core ideas of MIT scmutils:
  - A Lagrangian is a callable  L(t, q, qdot) -> expression
  - lagrange_equations(L) returns the Euler-Lagrange equations for a
    generalized coordinate treated as an unknown function of time.

All symbolic work is delegated to SymPy.
"""

import sympy as sp
from sympy import Function, Symbol, simplify, cos, sin, symbols


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
# Double-pendulum Lagrangian (5 symbolic parameters)
# ---------------------------------------------------------------------------

def L_double_pendulum(m1, m2, l1, l2, g):
    """
    Return a Lagrangian for a double pendulum with 5 symbolic parameters.

    The system consists of two point masses connected by rigid, massless
    rods.  Angles are measured from the vertical (downward).

    Parameters
    ----------
    m1, m2 : sympy symbols or numbers
        Masses of the first and second bobs.
    l1, l2 : sympy symbols or numbers
        Lengths of the first and second rods.
    g : sympy symbol or number
        Gravitational acceleration.

    Returns
    -------
    callable(t, qs, qdots) -> sympy expression
        Lagrangian suitable for ``lagrange_equations_multi``.
    """
    def L(t, qs, qdots):
        q1, q2 = qs
        qd1, qd2 = qdots

        # Kinetic energy
        T = (sp.Rational(1, 2) * (m1 + m2) * l1**2 * qd1**2
             + sp.Rational(1, 2) * m2 * l2**2 * qd2**2
             + m2 * l1 * l2 * qd1 * qd2 * cos(q1 - q2))

        # Potential energy (measured from pivot)
        V = -(m1 + m2) * g * l1 * cos(q1) - m2 * g * l2 * cos(q2)

        return T - V

    return L


# ---------------------------------------------------------------------------
# Legendre transform:  Lagrangian  ->  Hamiltonian
# ---------------------------------------------------------------------------

def legendre_transform(L_func, q_funcs, t: Symbol = None):
    """
    Compute the Hamiltonian from a Lagrangian via the Legendre transform.

    H(t, q, p) = Σ pᵢ q̇ᵢ − L(t, q, q̇)

    where pᵢ = ∂L/∂q̇ᵢ  are the conjugate momenta and q̇ᵢ are solved
    in terms of pᵢ.

    Parameters
    ----------
    L_func : callable
        Single-coordinate:  L(t, q, qdot) -> expr
        Multi-coordinate:   L(t, [q₁,…], [q̇₁,…]) -> expr
    q_funcs : Function or list of Function
        Generalized coordinate(s).
    t : Symbol, optional
        Independent variable (defaults to ``Symbol('t')``).

    Returns
    -------
    dict with keys:
        ``"H"``       – Hamiltonian expression in terms of (t, q, p)
        ``"momenta"`` – dict {pᵢ: ∂L/∂q̇ᵢ} (before solving)
        ``"qdots"``   – dict {q̇ᵢ: expr(p)} (velocities solved for momenta)
    """
    if t is None:
        t = Symbol("t")

    multi = isinstance(q_funcs, (list, tuple))
    if not multi:
        q_funcs = [q_funcs]

    qs = [qf(t) for qf in q_funcs]
    qdots = [q.diff(t) for q in qs]

    # Build L expression
    if multi or len(q_funcs) > 1:
        L_expr = L_func(t, qs, qdots)
    else:
        L_expr = L_func(t, qs[0], qdots[0])

    # Conjugate momenta: pᵢ = ∂L/∂q̇ᵢ
    p_symbols = symbols(
        " ".join(f"p_{qf.name}" for qf in q_funcs)
    )
    if len(q_funcs) == 1:
        p_symbols = [p_symbols]
    else:
        p_symbols = list(p_symbols)

    momentum_exprs = {p: L_expr.diff(qd) for p, qd in zip(p_symbols, qdots)}

    # Solve  pᵢ = ∂L/∂q̇ᵢ  for q̇ᵢ
    solve_eqs = [p - expr for p, expr in momentum_exprs.items()]
    sol = sp.solve(solve_eqs, qdots, dict=True)
    if not sol:
        raise ValueError("Could not solve for velocities in terms of momenta")
    qdot_of_p = sol[0]

    # H = Σ pᵢ q̇ᵢ − L,  then substitute q̇ -> q̇(p)
    H = sum(p * qd for p, qd in zip(p_symbols, qdots)) - L_expr
    H = H.subs(qdot_of_p)
    H = simplify(H)

    return {
        "H": H,
        "momenta": momentum_exprs,
        "qdots": qdot_of_p,
    }


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
