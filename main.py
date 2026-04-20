import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

a, b = 1.0, 4.0
I_exact = 21.0
J_exact = 31.0 / 5.0


def f_n(x, n):
    return np.floor(n * x ** 2) / n


def integrate_fn(n, measure_func):
    s = 0.0
    k0, k1 = int(np.floor(n * a * a)), int(np.floor(n * b * b))
    for k in range(k0, k1 + 1):
        l = max(a, np.sqrt(k / n))
        r = min(b, np.sqrt((k + 1) / n))
        s += f_n(l, n) * measure_func(l, r)
    return s


x = np.linspace(a, b, 4000)
for n in [1, 5, 20, 100]:
    plt.plot(x, f_n(x, n), label=f"n={n}")
plt.plot(x, x ** 2, "k--", label="f(x)=x^2")
plt.legend()
plt.grid(True)
plt.show()
tasks = [
    ("Lebesgue", I_exact, lambda l, r: r - l),
    ("Stieltjes", J_exact, lambda l, r: np.sqrt(r) - np.sqrt(l))
]

for name, exact, measure in tasks:
    print(f"{name}")
    for n_val in [10, 100, 1000]:
        t_start = perf_counter()
        val = integrate_fn(n_val, measure)
        dt = perf_counter() - t_start
        print(f"n={n_val:4d}: {val:.10f} | err={abs(val - exact):.2e} | time={dt:.4f}s")
    print()
