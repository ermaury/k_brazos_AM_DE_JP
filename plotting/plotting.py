"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from algorithms import Algorithm, EpsilonGreedy


def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    # elif isinstance(algo, OtroAlgoritmo):
    #     label += f" (parametro={algo.parametro})"
    # Añadir más condiciones para otros algoritmos aquí
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    plt.figure(figsize=(10, 6))
    # Generamos el eje x con los pasos de tiempo
    x_values = np.arange(1, steps + 1)

    # Graficamos cada algoritmo
    for i, algorithm in enumerate(algorithms):
        plt.plot(x_values, optimal_selections[i], label=get_algorithm_label(algorithm))

    # Configuración de la gráfica
    plt.xlabel("Pasos de tiempo")
    plt.ylabel("Porcentaje de selección del brazo óptimo")
    plt.title("Comparación de selección óptima entre algoritmos")
    plt.legend()
    plt.grid(True)
    
    # Mostrar la gráfica
    plt.show()


def plot_regret(steps: int, regrets: np.ndarray, algorithms: List[Algorithm], *args):
    """
    Genera la gráfica de Regret vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param regret: Matriz de regret (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param args: Parámetros adicionales, por ejemplo, la cota teórica (Cte * ln(T)).
    """
    plt.figure(figsize=(10, 6))

    # Eje x con los pasos de tiempo
    x_values = np.arange(1, steps + 1)

    # Graficar el regret para cada algoritmo
    for i, algorithm in enumerate(algorithms):
        plt.plot(x_values, regrets[i], label=get_algorithm_label(algorithm))

    # Si se proporciona una cota teórica (por ejemplo, C * ln(T)), la graficamos
    if args:
        for idx, bound in enumerate(args):
            plt.plot(x_values, bound, linestyle='dashed', label=f"Cota Teórica {idx+1}")

    # Configuración de la gráfica
    plt.xlabel("Pasos de tiempo")
    plt.ylabel("Regret")
    plt.title("Comparación del Regret entre Algoritmos")
    plt.legend()
    plt.grid(True)

    # Mostrar la gráfica
    plt.show()

