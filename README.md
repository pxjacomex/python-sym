# python-sym — Mecánica Lagrangiana Simbólica en Python

Port a Python del código de mecánica lagrangiana escrito en MIT Scheme con
[scmutils](https://groups.csail.mit.edu/mac/users/gjs/6946/linux-install.htm)
(el sistema simbólico del libro *Structure and Interpretation of Classical
Mechanics* de Sussman & Wisdom).

Toda la manipulación simbólica se delega a [SymPy](https://www.sympy.org/),
la biblioteca de álgebra computacional de Python.

## Código original (MIT Scheme / scmutils)

```scheme
(define ((L-pendulum m l g) local)
  (let ((theta     (coordinate local))
        (theta-dot (velocity local)))
    (- (* 1/2 m (expt (* l theta-dot) 2))
       (* m g l (- 1 (cos theta))))))

(show-expression
  (((Lagrange-equations (L-pendulum 'm 'l 'g))
    (literal-function 'theta))
   't))
```

## Equivalente en Python

```python
from sympy import symbols, Function, pprint
from python_sym import L_pendulum, lagrange_equations

m, l, g = symbols("m l g", positive=True)
theta = Function("theta")

L  = L_pendulum(m, l, g)
el = lagrange_equations(L, theta)

pprint(el, use_unicode=True)
# => g·l·m·sin(θ(t)) + l²·m·Derivative(θ(t), (t, 2))
```

Resultado: **m l² θ̈ + m g l sin(θ) = 0** — la ecuación del péndulo simple.

## Estructura del proyecto

```
python-sym/
├── python_sym/
│   ├── __init__.py          # Exportaciones públicas
│   └── lagrangian.py        # Euler-Lagrange, Lagrangianos, clase auxiliar
├── tests/
│   ├── test_symbolic_diff.py  # Tests de diferenciación simbólica (10 niveles)
│   └── test_lagrangian.py     # Tests de mecánica lagrangiana
├── pendulum_demo.py         # Demo ejecutable del péndulo
├── requirements.txt
└── README.md
```

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecutar la demo

```bash
python pendulum_demo.py
```

## Ejecutar los tests

```bash
pytest -v
```

### Cobertura de tests de diferenciación simbólica

Los tests en `tests/test_symbolic_diff.py` cubren 10 niveles de complejidad:

| Nivel | Tema | Ejemplos |
|-------|------|----------|
| 1 | Derivadas elementales | constantes, potencias, polinomios, derivadas de orden superior |
| 2 | Funciones trigonométricas | sin, cos, tan, sin², productos |
| 3 | Exponenciales y logaritmos | e^x, ln(x), x^x, cadenas |
| 4 | Regla de la cadena | sin(e^x), composiciones de 3+ niveles |
| 5 | Producto y cociente | producto triple, cocientes |
| 6 | Inversas e hiperbólicas | arcsin, arccos, arctan, sinh, cosh, tanh |
| 7 | Derivadas parciales | ∂/∂x, ∂/∂y, mixtas, Laplaciano |
| 8 | Funciones simbólicas genéricas | `literal-function` → `Function("f")` |
| 9 | Composiciones profundas | 5 niveles anidados, torre de potencias, atan2 |
| 10 | Propiedades de la diferenciación | linealidad, Leibniz, autofunciones de d/dx |

### Cobertura de tests lagrangianos

| Sistema | Qué se verifica |
|---------|-----------------|
| Péndulo simple | m l² θ̈ + m g l sin(θ) = 0 |
| Partícula libre | m ẍ = 0 |
| Oscilador armónico | m ẍ + k x = 0 |
| Péndulo doble | 2 ecuaciones E-L con derivadas segundas |
| Campo central | Ecuación radial con V'(r), conservación de momento angular |

## Correspondencia scmutils ↔ Python

| scmutils | Python (SymPy) |
|----------|----------------|
| `(literal-function 'theta)` | `Function("theta")` |
| `(coordinate local)` | argumento `q` del Lagrangiano |
| `(velocity local)` | argumento `qdot` del Lagrangiano |
| `(Lagrange-equations L)` | `lagrange_equations(L, q_func)` |
| `(show-expression ...)` | `pprint(expr, use_unicode=True)` |

## API

### `L_pendulum(m, l, g)`
Retorna un Lagrangiano `L(t, θ, θ̇)` para un péndulo simple.

### `lagrange_equations(L_func, q_func, t=None)`
Calcula la ecuación de Euler-Lagrange para una coordenada generalizada.

### `lagrange_equations_multi(L_func, q_funcs, t=None)`
Calcula las ecuaciones de Euler-Lagrange para múltiples coordenadas.

### `LagrangianMechanics(L_func, q_funcs, t=None)`
Clase auxiliar que agrupa un Lagrangiano con sus coordenadas y provee
`.euler_lagrange()` y `.show()`.
