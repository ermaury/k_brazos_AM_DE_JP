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

from typing import List, Dict

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


def plot_arm_statistics(arm_stats: List[Dict], algorithms: List, *args):
    """
    Genera gráficas de estadísticas de selección de brazos:
    Muestra el promedio de ganancias por brazo y el número de selecciones.
    
    :param arm_stats: Lista de diccionarios con estadísticas de cada brazo por algoritmo.
                      Cada diccionario debe contener:
                      - 'avg_rewards': np.array con promedio de recompensas por brazo.
                      - 'selections': np.array con cantidad de veces que se seleccionó cada brazo.
                      - 'optimal_arm': int con el índice del brazo óptimo.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param args: Parámetros adicionales que puedan ayudar en la visualización.
    """

    num_algorithms = len(algorithms)
    fig, axes = plt.subplots(1, num_algorithms, figsize=(6 * num_algorithms, 5))

    if num_algorithms == 1:
        axes = [axes]  # Asegurar que sea iterable si hay un solo algoritmo

    for idx, (algo, stats) in enumerate(zip(algorithms, arm_stats)):
        avg_rewards = stats['avg_rewards']
        selections = stats['selections']
        optimal_arm = stats['optimal_arm']
        num_arms = len(avg_rewards)

        # Etiquetas en el eje X: número del brazo + veces seleccionado
        labels = [f"Arm {i}\n({selections[i]:.0f} veces)" for i in range(num_arms)]

        # Colores: Resaltar el brazo óptimo
        colors = ['red' if i == optimal_arm else 'blue' for i in range(num_arms)]

        # Crear gráfico de barras
        axes[idx].bar(labels, avg_rewards, color=colors, alpha=0.7)

        # Asegurar que los ticks están configurados antes de asignar etiquetas
        axes[idx].set_xticks(range(len(labels)))  # Fijar posiciones de ticks
        axes[idx].set_xticklabels(labels, rotation=45, fontsize=10, ha="right")  # Aplicar etiquetas


        # Etiquetas y formato
        axes[idx].set_title(f"Estadísticas de brazos - {get_algorithm_label(algo)}")
        axes[idx].set_xlabel("Brazos (con número de selecciones)")
        axes[idx].set_ylabel("Promedio de recompensas")
        axes[idx].grid(axis='y', linestyle='--', alpha=0.6)

        # Agregar leyenda
        axes[idx].legend(["Óptimo" if i == optimal_arm else "No óptimo" for i in range(num_arms)], loc="best")

    plt.tight_layout()
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

