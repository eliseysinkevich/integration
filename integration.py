import math

import matplotlib.pyplot as plt
import numpy as np


def euler_integrator(h, y0, fun):
    """
    ДЛЯ ОДНОГО ШАГА
    y0 - начальное значение решения в момент времени t=0,
    h - шаг по времения,
    f(y) - правая часть дифференциального уравнения и его производная.
    Возвращают приближенное значение y(h)
    """
    return y0 + h * fun(y0)


def newton_integrator(h, y0, fun):
    return y0 + h * fun[0](y0) + fun[0](y0) * fun[1](y0) * h * h / 2


def modified_euler_integrator(h, y0, fun):
    """
    Сначала находим приближенное значение на половине шага
    """
    y_intermediate = y0 + fun(y0) * h / 2
    return y0 + h * fun(y_intermediate)


def runge_kutta_integrator(h, y0, fun):
    """
    Использует 4 вспомогательные точки
    """
    k1 = fun(y0)
    k2 = fun(y0 + k1 * h / 2)
    k3 = fun(y0 + k2 * h / 2)
    k4 = fun(y0 + k3 * h)
    return y0 + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6


def newton_method(functions, x0):
    """
    Для неявного метода Эйлера
    Находит решение уравнения f(x)=0 методом Ньютона.
    x0 - начальное приближение.
    fun=(f(x),dF(x)) - функция и ее производная.
    Возвращает решение уравнения.
    """
    for i in range(100):  # ограничиваем максимальное число итераций
        x = x0 - functions[0](x0) / functions[1](x0)
        if x == x0:  # достигнута максимальная точность
            break
        x0 = x
    return x0


def backward_euler_integrator(h, y0, fun):
    functions = (lambda y: y0 + h * fun[0](y) - y, lambda y: h * fun[1](y) - 1)
    return newton_method(functions, y0)


def integrate(n, delta, fun, y0, integrator):
    """
    ДЛЯ ИНТЕРВАЛА
    Делает N шагов длины delta метода integrator для уравнения y'=f(y) с начальными условиями y0.
    Возвращает значение решения в конце интервала.
    """
    for n in range(n):
        y0 = integrator(delta, y0, fun)
    return y0


def interval_error_plot(fun, y, integrator, t=1, max_number_of_steps=1000, number_of_points_on_plot=16):
    """
    График зависимости погрешности интегрирования на интервале от длины шага интегрирования.
    """
    eps = np.finfo(float).eps
    number_of_steps = np.logspace(0, np.log10(max_number_of_steps), number_of_points_on_plot).astype(
        np.int)  # количество шагов (массив из 16 элементов)
    steps = t / number_of_steps  # длина шагов интегрирования: длина интервала/количество шагов
    y0 = y(0)  # начальное значение
    y_precise = y(t)  # точное значения решения на правом конце
    #    print (y_precise)
    y_approximate = [integrate(N, t / N, fun, y0, integrator) for N in number_of_steps]  # приближенные решения
    #    print(y_approximate)
    h = [np.maximum(np.max(np.abs(y_precise - ya)), eps) for ya in y_approximate]  # погрешность
    plt.loglog(steps, h, '.-')
    plt.xlabel("Шаг интегрирования")
    plt.ylabel("Погрешность интегрования на интервале")


def order_plot(order):
    """
    ПОРЯДКИ
    Рисует на текущем графике кривую y = x ** order.
    """
    ax = plt.gca()
    steps = np.asarray(ax.get_xlim())
    plt.loglog(steps, steps ** order, '--r')


def y_exact(t):
    """
    Аналитическое решение дифференциального уравнения
    """
    return 2 * math.atan(math.tanh(t / 2 + math.atanh(math.tan(0.5))))


# Уравнение.
# f[0] - уравнение, f[1] - производная
f = (lambda y: math.cos(y), lambda y: -math.sin(y))

plt.figure()
interval_error_plot(f[0], y_exact, euler_integrator)
order_plot(1)
plt.legend(["метод Эйлера", "Первый порядок"], loc=2)
plt.show()

plt.figure()
interval_error_plot(f[0], y_exact, modified_euler_integrator)
order_plot(2)
plt.legend(["Модифицированный метод Эйлера", "Второй порядок"], loc=2)
plt.show()

plt.figure()
interval_error_plot(f, y_exact, newton_integrator)
order_plot(2)
plt.legend(["Метод Ньютона", "Второй порядок"], loc=2)
plt.show()

plt.figure()
interval_error_plot(f[0], y_exact, runge_kutta_integrator)
order_plot(4)
plt.legend(["Метод Рунге-Кутты", "Четвертый порядок"], loc=2)
plt.show()

plt.figure()
interval_error_plot(f, y_exact, backward_euler_integrator)
order_plot(1)
plt.legend(["Неявный метод Эйлера", "Первый порядок"], loc=2)
plt.show()
