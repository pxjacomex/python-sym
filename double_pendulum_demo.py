#!/usr/bin/env python3
"""
DEMO 2: Pendulo doble — Lagrangiano y Hamiltoniano
====================================================

Sistema de dos masas (m1, m2) conectadas por barras rigidas de
longitudes (l1, l2) bajo gravedad g.  Cinco parametros simbolicos.

Se muestra:
  1. El Lagrangiano L(t, q, qdot)
  2. Las ecuaciones de Euler-Lagrange
  3. Los momentos conjugados
  4. El Hamiltoniano obtenido por transformada de Legendre
"""

import sympy as sp
from sympy import Function, Symbol, pprint, symbols

from python_sym import (
    L_double_pendulum,
    lagrange_equations_multi,
    legendre_transform,
)

# ---------- Simbolos ----------
m1, m2, l1, l2, g = symbols("m1 m2 l1 l2 g", positive=True)
t = Symbol("t")
theta1 = Function("theta1")
theta2 = Function("theta2")

# ---------- Lagrangiano ----------
L = L_double_pendulum(m1, m2, l1, l2, g)

q1, q2 = theta1(t), theta2(t)
qd1, qd2 = q1.diff(t), q2.diff(t)
L_expr = L(t, [q1, q2], [qd1, qd2])

print("=" * 65)
print("  Pendulo doble — 5 parametros simbolicos (m1, m2, l1, l2, g)")
print("=" * 65)
print()
print("Lagrangiano  L = T - V :")
pprint(L_expr, use_unicode=True)
print()

# ---------- Ecuaciones de Euler-Lagrange ----------
print("-" * 65)
print("Ecuaciones de Euler-Lagrange")
print("-" * 65)
eqs = lagrange_equations_multi(L, [theta1, theta2], t)
for i, eq in enumerate(eqs, 1):
    print(f"\n  EL({i}) = 0 :")
    pprint(eq, use_unicode=True)
print()

# ---------- Transformada de Legendre -> Hamiltoniano ----------
print("-" * 65)
print("Transformada de Legendre:  L  ->  H")
print("-" * 65)

result = legendre_transform(L, [theta1, theta2], t)

print("\nMomentos conjugados:")
for p_sym, expr in result["momenta"].items():
    print(f"  {p_sym} = ", end="")
    pprint(expr, use_unicode=True)

print("\nVelocidades en terminos de momentos:")
for qd, expr in result["qdots"].items():
    pprint(sp.Eq(qd, expr), use_unicode=True)
    print()

print("Hamiltoniano  H(t, q, p) :")
pprint(result["H"], use_unicode=True)
print()

# ---------- Verificacion: H = T + V para sistema natural ----------
print("-" * 65)
print("Verificacion:  H debe ser T + V  (sistema natural)")
print("-" * 65)

T = (sp.Rational(1, 2) * (m1 + m2) * l1**2 * qd1**2
     + sp.Rational(1, 2) * m2 * l2**2 * qd2**2
     + m2 * l1 * l2 * qd1 * qd2 * sp.cos(q1 - q2))
V = -(m1 + m2) * g * l1 * sp.cos(q1) - m2 * g * l2 * sp.cos(q2)
E = T + V
E_in_p = E.subs(result["qdots"])
E_in_p = sp.simplify(E_in_p)

diff_check = sp.simplify(result["H"] - E_in_p)
if diff_check == 0:
    print("  OK  H = T + V  (coincide exactamente)")
else:
    print(f"  Diferencia residual: {diff_check}")

print()
print("=" * 65)
