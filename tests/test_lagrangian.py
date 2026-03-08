"""
Tests para el módulo de mecánica lagrangiana.

Cubre:
  - Péndulo simple (port directo del código scmutils)
  - Partícula libre
  - Oscilador armónico
  - Péndulo doble (multi-coordenada)
  - Partícula en campo central
"""

import sympy as sp
from sympy import (
    Symbol, Function, Rational, symbols,
    sin, cos, simplify,
)

from python_sym.lagrangian import (
    lagrange_equations,
    lagrange_equations_multi,
    L_pendulum,
    LagrangianMechanics,
)

t = Symbol("t")


# ===================================================================
#  Péndulo simple — Port directo del código scmutils
# ===================================================================

class TestPendulum:

    def test_euler_lagrange_equation(self):
        """
        El resultado debe ser:  m l² θ̈ + m g l sin(θ) = 0
        (equivalente a la salida de show-expression en scmutils)
        """
        m, l, g = symbols("m l g", positive=True)
        theta = Function("theta")

        L = L_pendulum(m, l, g)
        el = lagrange_equations(L, theta, t)

        theta_t = theta(t)
        theta_ddot = theta_t.diff(t, 2)
        expected = m * l**2 * theta_ddot + m * g * l * sin(theta_t)

        assert simplify(el - expected) == 0

    def test_numeric_parameters(self):
        """Con parámetros numéricos m=1, l=1, g=10."""
        theta = Function("theta")
        L = L_pendulum(1, 1, 10)
        el = lagrange_equations(L, theta, t)

        theta_t = theta(t)
        expected = theta_t.diff(t, 2) + 10 * sin(theta_t)
        assert simplify(el - expected) == 0

    def test_via_class(self):
        """Verificar la interfaz OOP."""
        m, l, g = symbols("m l g", positive=True)
        theta = Function("theta")
        L = L_pendulum(m, l, g)
        mech = LagrangianMechanics(L, theta, t)
        eqs = mech.euler_lagrange()
        assert len(eqs) == 1
        assert eqs[0] != 0  # no trivial


# ===================================================================
#  Partícula libre
# ===================================================================

class TestFreeParticle:

    def test_free_particle_1d(self):
        """L = ½ m ẋ²  =>  m ẍ = 0"""
        m_sym = Symbol("m", positive=True)
        x_func = Function("x")

        def L_free(t_val, x_val, xdot):
            return Rational(1, 2) * m_sym * xdot**2

        el = lagrange_equations(L_free, x_func, t)
        expected = m_sym * x_func(t).diff(t, 2)
        assert simplify(el - expected) == 0


# ===================================================================
#  Oscilador armónico
# ===================================================================

class TestHarmonicOscillator:

    def test_harmonic_oscillator(self):
        """L = ½ m ẋ² - ½ k x²  =>  m ẍ + k x = 0"""
        m_sym, k = symbols("m k", positive=True)
        x_func = Function("x")

        def L_ho(t_val, x_val, xdot):
            return Rational(1, 2) * m_sym * xdot**2 - Rational(1, 2) * k * x_val**2

        el = lagrange_equations(L_ho, x_func, t)
        x_t = x_func(t)
        expected = m_sym * x_t.diff(t, 2) + k * x_t
        assert simplify(el - expected) == 0


# ===================================================================
#  Péndulo doble (multi-coordenada)
# ===================================================================

class TestDoublePendulum:

    def test_double_pendulum_structure(self):
        """
        Verificar que genera 2 ecuaciones E-L para un péndulo doble.
        L = T₁ + T₂ - V₁ - V₂
        """
        m1, m2, l1, l2, g_sym = symbols("m1 m2 l1 l2 g", positive=True)
        th1 = Function("theta1")
        th2 = Function("theta2")

        def L_double(t_val, qs, qdots):
            q1, q2 = qs
            qd1, qd2 = qdots

            # Velocidades cartesianas expresadas con qdots
            vx1 = l1 * cos(q1) * qd1
            vy1 = l1 * sin(q1) * qd1
            vx2 = vx1 + l2 * cos(q2) * qd2
            vy2 = vy1 + l2 * sin(q2) * qd2

            T = Rational(1, 2) * m1 * (vx1**2 + vy1**2) + \
                Rational(1, 2) * m2 * (vx2**2 + vy2**2)
            V = -m1 * g_sym * l1 * cos(q1) - m2 * g_sym * (l1 * cos(q1) + l2 * cos(q2))
            return T - V

        eqs = lagrange_equations_multi(L_double, [th1, th2], t)
        assert len(eqs) == 2
        # Both equations should contain second derivatives
        for eq in eqs:
            assert eq.has(sp.Derivative)


# ===================================================================
#  Campo central
# ===================================================================

class TestCentralForce:

    def test_central_force_radial(self):
        """
        L = ½ m (ṙ² + r² φ̇²) - V(r)
        Ecuación radial: m r̈ - m r φ̇² + V'(r) = 0
        """
        m_sym = Symbol("m", positive=True)
        r = Function("r")
        phi = Function("phi")
        V = Function("V")

        def L_central(t_val, qs, qdots):
            r_val, phi_val = qs
            rdot, phidot = qdots
            T = Rational(1, 2) * m_sym * (rdot**2 + r_val**2 * phidot**2)
            return T - V(r_val)

        eqs = lagrange_equations_multi(L_central, [r, phi], t)
        assert len(eqs) == 2

        # The angular equation should give conservation of angular momentum:
        # d/dt(m r² φ̇) = 0  =>  the EL eq for phi
        # The radial equation should contain V'(r)
        radial_eq = eqs[0]
        assert radial_eq.has(sp.Derivative)

    def test_angular_momentum_conservation(self):
        """
        Para V que no depende de φ, la ecuación de φ
        es d/dt(m r² φ̇) = 0.
        """
        m_sym = Symbol("m", positive=True)
        k = Symbol("k", positive=True)
        r = Function("r")
        phi = Function("phi")

        def L_central(t_val, qs, qdots):
            r_val, phi_val = qs
            rdot, phidot = qdots
            T = Rational(1, 2) * m_sym * (rdot**2 + r_val**2 * phidot**2)
            V = -k / r_val  # gravitacional / Coulomb
            return T - V

        eqs = lagrange_equations_multi(L_central, [r, phi], t)
        phi_eq = eqs[1]

        # La ecuación de phi no debe contener phi(t) explícitamente
        # (solo derivadas), reflejando la simetría rotacional.
        # Concretamente: d/dt(m r² φ̇) = 0
        # Expandido: m (2 r ṙ φ̇ + r² φ̈) = 0
        r_t = r(t)
        phi_t = phi(t)
        expected = m_sym * (
            2 * r_t.diff(t) * r_t * phi_t.diff(t)
            + r_t**2 * phi_t.diff(t, 2)
        )
        assert simplify(phi_eq - expected) == 0
