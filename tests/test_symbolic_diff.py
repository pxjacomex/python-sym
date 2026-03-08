"""
Tests de diferenciación simbólica — de lo más simple a lo más complejo.

Verifica que SymPy produce las derivadas correctas para una progresión
de funciones que va desde polinomios triviales hasta composiciones
profundamente anidadas.
"""

import pytest
import sympy as sp
from sympy import (
    Symbol, Function, Rational,
    sin, cos, tan, exp, log, sqrt, atan2,
    asin, acos, atan, sinh, cosh, tanh,
    pi, oo, simplify, diff, symbols,
)

x, y, z, a, b, c, t = symbols("x y z a b c t")


# ===================================================================
#  NIVEL 1 — Derivadas elementales
# ===================================================================

class TestElementaryDerivatives:
    """Polinomios, potencias y constantes."""

    def test_constant(self):
        assert diff(5, x) == 0

    def test_symbol(self):
        assert diff(x, x) == 1

    def test_linear(self):
        assert diff(3 * x + 7, x) == 3

    def test_power(self):
        assert diff(x**4, x) == 4 * x**3

    def test_rational_power(self):
        assert diff(x ** Rational(3, 2), x) == Rational(3, 2) * sqrt(x)

    def test_negative_power(self):
        assert diff(1 / x, x) == -1 / x**2

    def test_polynomial(self):
        f = 2 * x**3 - 5 * x**2 + 4 * x - 1
        assert diff(f, x) == 6 * x**2 - 10 * x + 4

    def test_second_derivative(self):
        assert diff(x**5, x, 2) == 20 * x**3

    def test_third_derivative(self):
        assert diff(x**6, x, 3) == 120 * x**3


# ===================================================================
#  NIVEL 2 — Funciones trigonométricas
# ===================================================================

class TestTrigDerivatives:

    def test_sin(self):
        assert diff(sin(x), x) == cos(x)

    def test_cos(self):
        assert diff(cos(x), x) == -sin(x)

    def test_tan(self):
        assert simplify(diff(tan(x), x) - (1 + tan(x)**2)) == 0

    def test_sin_squared(self):
        assert simplify(diff(sin(x)**2, x) - 2 * sin(x) * cos(x)) == 0

    def test_cos_squared(self):
        assert simplify(diff(cos(x)**2, x) + 2 * sin(x) * cos(x)) == 0

    def test_sin_cos_product(self):
        # d/dx [sin(x) cos(x)] = cos²x - sin²x = cos(2x)
        d = diff(sin(x) * cos(x), x)
        assert simplify(d - sp.cos(2 * x)) == 0


# ===================================================================
#  NIVEL 3 — Exponenciales y logaritmos
# ===================================================================

class TestExpLogDerivatives:

    def test_exp(self):
        assert diff(exp(x), x) == exp(x)

    def test_log(self):
        assert diff(log(x), x) == 1 / x

    def test_exp_chain(self):
        assert diff(exp(3 * x), x) == 3 * exp(3 * x)

    def test_log_chain(self):
        assert simplify(diff(log(x**2 + 1), x) - 2 * x / (x**2 + 1)) == 0

    def test_x_to_the_x(self):
        # d/dx [x^x] = x^x (ln x + 1)
        f = x**x
        d = diff(f, x)
        expected = x**x * (log(x) + 1)
        assert simplify(d - expected) == 0


# ===================================================================
#  NIVEL 4 — Regla de la cadena (composiciones)
# ===================================================================

class TestChainRule:

    def test_sin_of_exp(self):
        assert diff(sin(exp(x)), x) == cos(exp(x)) * exp(x)

    def test_exp_of_sin(self):
        assert diff(exp(sin(x)), x) == exp(sin(x)) * cos(x)

    def test_log_of_sin(self):
        assert simplify(diff(log(sin(x)), x) - cos(x) / sin(x)) == 0

    def test_nested_three_levels(self):
        # d/dx sin(cos(exp(x)))
        f = sin(cos(exp(x)))
        d = diff(f, x)
        expected = cos(cos(exp(x))) * (-sin(exp(x))) * exp(x)
        assert simplify(d - expected) == 0

    def test_sqrt_of_polynomial(self):
        f = sqrt(x**2 + 1)
        d = diff(f, x)
        expected = x / sqrt(x**2 + 1)
        assert simplify(d - expected) == 0


# ===================================================================
#  NIVEL 5 — Regla del producto y cociente
# ===================================================================

class TestProductQuotient:

    def test_product_rule(self):
        f = x**2 * sin(x)
        d = diff(f, x)
        expected = 2 * x * sin(x) + x**2 * cos(x)
        assert simplify(d - expected) == 0

    def test_quotient_rule(self):
        f = sin(x) / x
        d = diff(f, x)
        expected = (x * cos(x) - sin(x)) / x**2
        assert simplify(d - expected) == 0

    def test_triple_product(self):
        f = x * sin(x) * exp(x)
        d = diff(f, x)
        expected = sin(x) * exp(x) + x * cos(x) * exp(x) + x * sin(x) * exp(x)
        assert simplify(d - expected) == 0


# ===================================================================
#  NIVEL 6 — Funciones trigonométricas inversas e hiperbólicas
# ===================================================================

class TestInverseAndHyperbolic:

    def test_asin(self):
        d = diff(asin(x), x)
        expected = 1 / sqrt(1 - x**2)
        assert simplify(d - expected) == 0

    def test_acos(self):
        d = diff(acos(x), x)
        expected = -1 / sqrt(1 - x**2)
        assert simplify(d - expected) == 0

    def test_atan(self):
        d = diff(atan(x), x)
        expected = 1 / (1 + x**2)
        assert simplify(d - expected) == 0

    def test_sinh(self):
        assert diff(sinh(x), x) == cosh(x)

    def test_cosh(self):
        assert diff(cosh(x), x) == sinh(x)

    def test_tanh(self):
        d = diff(tanh(x), x)
        expected = 1 - tanh(x)**2
        assert simplify(d - expected) == 0


# ===================================================================
#  NIVEL 7 — Derivadas parciales (multivariable)
# ===================================================================

class TestPartialDerivatives:

    def test_partial_x(self):
        f = x**2 * y + y**3
        assert diff(f, x) == 2 * x * y

    def test_partial_y(self):
        f = x**2 * y + y**3
        assert diff(f, y) == x**2 + 3 * y**2

    def test_mixed_partial(self):
        f = x**2 * y**3
        assert diff(f, x, y) == 6 * x * y**2

    def test_symmetry_of_mixed(self):
        f = sin(x * y) + exp(x + y)
        assert simplify(diff(f, x, y) - diff(f, y, x)) == 0

    def test_laplacian_2d(self):
        # ∇²(x² + y²) = 2 + 2 = 4
        f = x**2 + y**2
        laplacian = diff(f, x, 2) + diff(f, y, 2)
        assert laplacian == 4


# ===================================================================
#  NIVEL 8 — Funciones simbólicas genéricas (literal-function)
# ===================================================================

class TestLiteralFunctions:
    """Análogo a literal-function de scmutils: funciones sin definición
    explícita cuyas derivadas quedan en forma simbólica."""

    def test_basic_derivative(self):
        f = Function("f")
        d = diff(f(t), t)
        # Debe quedar como Derivative(f(t), t)
        assert d == f(t).diff(t)

    def test_chain_with_literal(self):
        f = Function("f")
        g = Function("g")
        # d/dt f(g(t)) = f'(g(t)) * g'(t)
        d = diff(f(g(t)), t)
        assert d == sp.Subs(
            sp.Derivative(f(x), x), x, g(t)
        ).doit() * g(t).diff(t) or simplify(
            d - sp.Derivative(f(g(t)), g(t)) * sp.Derivative(g(t), t)
        ) == 0

    def test_second_derivative_literal(self):
        f = Function("f")
        d2 = diff(f(t), t, 2)
        assert d2 == sp.Derivative(f(t), (t, 2))

    def test_product_with_literal(self):
        f = Function("f")
        d = diff(t * f(t), t)
        expected = f(t) + t * f(t).diff(t)
        assert simplify(d - expected) == 0


# ===================================================================
#  NIVEL 9 — Composiciones profundas y expresiones complejas
# ===================================================================

class TestComplexCompositions:

    def test_nested_exp_sin_log(self):
        f = exp(sin(log(x)))
        d = diff(f, x)
        expected = exp(sin(log(x))) * cos(log(x)) / x
        assert simplify(d - expected) == 0

    def test_power_tower(self):
        # d/dx [x^(x^x)]
        f = x ** (x**x)
        d = diff(f, x)
        # Just check it's non-zero and evaluates at x=2
        val = d.subs(x, 2).evalf()
        assert abs(val) > 0

    def test_atan2_derivative(self):
        f = atan2(y, x)
        dx = diff(f, x)
        dy = diff(f, y)
        # ∂/∂x atan2(y,x) = -y/(x²+y²)
        assert simplify(dx + y / (x**2 + y**2)) == 0
        # ∂/∂y atan2(y,x) = x/(x²+y²)
        assert simplify(dy - x / (x**2 + y**2)) == 0

    def test_implicit_differentiation_style(self):
        # f(x,y) = x² + y² - 1,  dy/dx = -x/y  (circle)
        f = x**2 + y**2 - 1
        dydx = -diff(f, x) / diff(f, y)
        assert simplify(dydx + x / y) == 0

    def test_leibniz_formula(self):
        # n-th derivative of product: verify for n=3, f=x^3, g=exp(x)
        f = x**3
        g = exp(x)
        # Direct
        d3_fg = diff(f * g, x, 3)
        # Leibniz: sum C(3,k) f^(k) g^(3-k)
        leibniz = sum(
            sp.binomial(3, k) * diff(f, x, k) * diff(g, x, 3 - k)
            for k in range(4)
        )
        assert simplify(d3_fg - leibniz) == 0

    def test_five_level_nesting(self):
        # sin(exp(cos(log(tan(x)))))
        f = sin(exp(cos(log(tan(x)))))
        d = diff(f, x)
        # Verify numerically at x = 1.0 (where tan > 0)
        val = complex(d.subs(x, 1.0).evalf())
        assert abs(val) > 0  # just verify it computes


# ===================================================================
#  NIVEL 10 — Identidades y propiedades de la diferenciación
# ===================================================================

class TestDifferentiationProperties:

    def test_linearity(self):
        f = sin(x)
        g = cos(x)
        assert diff(3 * f + 5 * g, x) == 3 * diff(f, x) + 5 * diff(g, x)

    def test_product_rule_identity(self):
        f = x**2
        g = exp(x)
        lhs = diff(f * g, x)
        rhs = diff(f, x) * g + f * diff(g, x)
        assert simplify(lhs - rhs) == 0

    def test_chain_rule_identity(self):
        f = Function("f")
        g = Function("g")
        # d/dx f(g(x)) should use chain rule
        d = diff(f(g(x)), x)
        # Must contain g'(x) as factor
        assert g(x).diff(x) in d.atoms(sp.Derivative) or d.has(
            sp.Derivative(g(x), x)
        )

    def test_higher_order_sin(self):
        # Fourth derivative of sin = sin
        assert simplify(diff(sin(x), x, 4) - sin(x)) == 0

    def test_higher_order_cos(self):
        # Fourth derivative of cos = cos
        assert simplify(diff(cos(x), x, 4) - cos(x)) == 0

    def test_exp_is_eigenfunction(self):
        # n-th derivative of e^x is e^x, for any n
        for n in range(1, 6):
            assert diff(exp(x), x, n) == exp(x)
