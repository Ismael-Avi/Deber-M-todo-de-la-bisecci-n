import numpy as np
import matplotlib.pyplot as plt

C1 = 'blue'
C2 = 'red'
CG = 'gray'
CP = 'black'

sol_x = 4.493405
eps = 0.01

x = np.linspace(0, 5, 500)
plt.plot(x, x, color=C1, label='y = x')

x1 = np.linspace(0, np.pi/2 - eps, 100)
x2 = np.linspace(np.pi/2 + eps, 3*np.pi/2 - eps, 200)
x3 = np.linspace(3*np.pi/2 + eps, 5, 100)

plt.plot(x1, np.tan(x1), color=C2, label='y = tan(x)')
plt.plot(x2, np.tan(x2), color=C2)
plt.plot(x3, np.tan(x3), color=C2)

plt.axvline(np.pi/2, color=CG, linestyle='--', label='Asíntotas de tan(x)')
plt.axvline(3*np.pi/2, color=CG, linestyle='--')

plt.plot(sol_x, sol_x, 'o', color=CP, label=f'Solución: x = {sol_x:.6f}')

plt.xlim(0, 5)
plt.ylim(-10, 10)
plt.xlabel('Eje x')
plt.ylabel('Eje y')
plt.title('Gráfica de y = x e y = tan(x)')
plt.legend()
plt.grid()
plt.show()


def biseccion(f, a, b, TOL=1e-6, N0=100):
    """Devuelve (p, historia), con historia = [(a, b, p) por iteración]."""
    FA = f(a)
    historia = []
    i = 1

    while i <= N0:
        p = (a + b) / 2
        FP = f(p)
        historia.append((a, b, p))

        if FP == 0 or (b - a) / 2 < TOL:
            return p, historia

        i += 1

        if FA * FP > 0:
            a = p
            FA = FP
        else:
            b = p

    raise RuntimeError(f"Fracaso después de {N0} iteraciones.")


grav = 9.81
s0, m, k = 300.0, 0.25, 0.1


def s(t):
    return s0 - (m * grav / k) * t + (m ** 2 * grav / k ** 2) * (1 - np.exp(-k * t / m))


def graficar(t_star, historia, k_iteracion=None, archivo="caida.png"):
    """Dibuja s(t). Si k_iteracion no es None, resalta el intervalo [a, b]
    de esa iteración (1-based) y su punto medio p."""
    t = np.linspace(0, 20, 400)
    plt.figure(figsize=(9, 4.5))
    plt.plot(t, s(t), color=C1, label="s(t)")
    plt.axhline(0, color=CG, linestyle="--", linewidth=1)
    if k_iteracion is not None:
        a, b, p = historia[k_iteracion - 1]
        plt.axvspan(a, b, color=CG, alpha=0.25,
                    label=f"iter {k_iteracion}: [{a:.4g}, {b:.4g}]")
        plt.axvline(p, color=C2, linestyle=":", linewidth=1.5,
                    label=f"p = {p:.4g}")
    plt.plot(t_star, 0, "o", color=CP, label=f"t* = {t_star:.6f}")
    plt.xlabel("t (s)")
    plt.ylabel("s(t) (m)")
    plt.title("Caída con resistencia del aire")
    plt.legend()
    plt.grid(True)
    plt.savefig(archivo, dpi=150)
    plt.show()


t_star, historia = biseccion(s, 0.0, 20.0, TOL=1e-4)
graficar(t_star, historia, k_iteracion=3)


def f(x):
    return (x + 3) * (x + 1)**2 * x * (x - 1)**3 * (x - 3)


intervalos = [
    (-3.5, 2.5),
    (-1.5, 0.6),
    (0.4, 2.2),
    (2.0, 3.7)
]

for a, b in intervalos:
    p, historia = biseccion(f, a, b, TOL=1e-6)
    print(f"Intervalo [{a}, {b}] -> Converge al cero: {round(p)}")


L = 10.0
r = 1.0
V = 12.4


def f_abrevadero(h):
    return L * (0.5 * np.pi * r**2 - r**2 * np.arcsin(h / r) - h * np.sqrt(r**2 - h**2)) - V


h_aprox, historia = biseccion(f_abrevadero, 0.0, 1.0, TOL=0.01)
profundidad = r - h_aprox

print(f"Iteraciones realizadas: {len(historia)}")
print(f"Distancia h encontrada: {h_aprox:.6f} cm (o {h_aprox:.2f} cm)")
print(f"Profundidad del agua:   {profundidad:.6f} cm (o {profundidad:.2f} cm)\n")

print(f"{'i':>2} | {'a':>8} | {'b':>8} | {'p (h)':>8} | {'(b-a)/2':>8}")
print("-" * 45)
for i, (a_i, b_i, p_i) in enumerate(historia, 1):
    print(f"{i:2d} | {a_i:8.4f} | {b_i:8.4f} | {p_i:8.4f} | {(b_i - a_i)/2:8.4f}")


x = np.linspace(-2.2, 0.2, 300)
plt.figure(figsize=(7, 4))
plt.plot(x, x**2 - 1, color=C1, label=r"$y = x^2 - 1$")
plt.plot(x, np.exp(1 - x**2), color=C2, label=r"$y = e^{1 - x^2}$")
plt.axvspan(-2.0, 0.0, color=CG, alpha=0.2, label="Intervalo [-2, 0]")
plt.title(r"Intersección de $y = x^2 - 1$ y $y = e^{1-x^2}$")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()


def w(x):
    return (x**2 - 1) - np.exp(1 - x**2)


p, historia = biseccion(w, -2.0, 0.0, TOL=1e-3)
print(f"Solución encontrada: {p:.6f}")


x = np.linspace(3.5, 4.65, 300)
plt.figure(figsize=(7, 4))
plt.plot(x, x, color=C1, label=r"$y = x$")
plt.plot(x, np.tan(x), color=C2, label=r"$y = \tan(x)$")
plt.axvspan(4.0, 4.6, color=CG, alpha=0.2, label="Intervalo elegido [4.0, 4.6]")
plt.title(r"Intersección de $y = x$ y $y = \tan(x)$")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()


def Z(x):
    return x - np.tan(x)


p, historia = biseccion(Z, 4.0, 4.6, TOL=1e-5)
print(f"Solución encontrada: {p:.6f}")


x = np.linspace(0, 3, 300)
plt.figure(figsize=(7, 4))
plt.plot(x, x, color=C1, label=r"$y = x$")
plt.plot(x, 2 * np.sin(x), color=C2, label=r"$y = 2\sin(x)$")
plt.axvspan(1.0, 2.0, color=CG, alpha=0.2, label="Intervalo elegido [1, 2]")
plt.title(r"Intersección de $y = x$ y $y = 2\sin(x)$")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()


def R(x):
    return x - 2 * np.sin(x)


p, historia = biseccion(R, 1.0, 2.0, TOL=1e-5)
print(f"Solución encontrada: {p:.6f}")


def g(x):
    return x**3 - 7*x**2 + 14*x - 6


p1, historia1 = biseccion(g, 0.0, 1.0, TOL=1e-2)
p2, historia2 = biseccion(g, 1.0, 3.2, TOL=1e-2)
p3, historia3 = biseccion(g, 3.2, 4.0, TOL=1e-2)

print(f"Solución encontrada en [0,1]: {p1:.6f}")
print(f"Solución encontrada en [1,3.2]: {p2:.6f}")
print(f"Solución encontrada en [3.2,4]: {p3:.6f}")
print()

encabezado = f"{'i':>3} | {'a':>10} | {'b':>10} | {'p':>10} | {'g(p)':>12} | {'(b-a)/2':>10}"
for h in (historia1, historia2, historia3):
    print(encabezado)
    print("-" * len(encabezado))
    for i, (a_i, b_i, p_i) in enumerate(h, 1):
        print(f"{i:3d} | {a_i:10.6f} | {b_i:10.6f} | {p_i:10.6f} | "
              f"{g(p_i):12.4e} | {(b_i - a_i)/2:10.6f}")
    print()
