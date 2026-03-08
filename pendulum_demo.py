#!/usr/bin/env python3
"""
DEMO 1: Péndulo simple — Ecuaciones de Euler-Lagrange automáticas
=================================================================

Port directo del código scmutils:

    (define ((L-pendulum m l g) local)
      (let ((theta     (coordinate local))
            (theta-dot (velocity local)))
        (- (* 1/2 m (expt (* l theta-dot) 2))
           (* m g l (- 1 (cos theta))))))

    (show-expression
      (((Lagrange-equations (L-pendulum 'm 'l 'g))
        (literal-function 'theta))
       't))
"""

import sympy as sp
from sympy import Function, Symbol, pprint

from python_sym import L_pendulum, lagrange_equations

# ---------- Símbolos ----------
m, l, g = sp.symbols("m l g", positive=True)
t = Symbol("t")
theta = Function("theta")            # literal-function 'theta

# ---------- Lagrangiano ----------
L = L_pendulum(m, l, g)

# ---------- Ecuaciones de Euler-Lagrange ----------
el_eq = lagrange_equations(L, theta, t)

print("=" * 60)
print("Péndulo simple — Ecuación de Euler-Lagrange")
print("=" * 60)
print()
print("L = T - V = ½ m l² θ̇²  -  m g l (1 - cos θ)")
print()
print("Ecuación de Euler-Lagrange  (= 0):")
pprint(el_eq, use_unicode=True)
print()

# Verificar que la ecuación es:  m l² θ̈ + m g l sin(θ) = 0
theta_t = theta(t)
theta_ddot = theta_t.diff(t, 2)

expected = m * l**2 * theta_ddot + m * g * l * sp.sin(theta_t)
diff_check = sp.simplify(el_eq - expected)

print("Verificación contra  m l² θ̈ + m g l sin(θ):")
if diff_check == 0:
    print("  ✓  Coincide exactamente.")
else:
    print(f"  Diferencia residual: {diff_check}")

print()
print("=" * 60)
