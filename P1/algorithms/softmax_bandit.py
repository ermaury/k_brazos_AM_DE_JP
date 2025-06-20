"""
Module: algorithms/softmax_bandit.py
Description: Implementación del algoritmo Softmax (Boltzmann Exploration)
para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/05/13

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np
from algorithms.algorithm import Algorithm

class SoftmaxBandit(Algorithm):
    def __init__(self, k: int, tau: float = 0.1):
        """
        Inicializa el algoritmo Softmax clásico (Boltzmann exploration).

        :param k: Número de brazos.
        :param tau: Temperatura para suavizar la probabilidad.
        """
        super().__init__(k)
        assert tau > 0, "La temperatura tau debe ser positiva."
        self.tau = tau
        self.values = np.zeros(k)    # Aquí guardamos las estimaciones de recompensa media Q(a)
        self.counts = np.zeros(k)    # Número de veces que hemos jugado cada brazo

    def get_algorithm_label(self) -> str:
        label = type(self).__name__ + f" (tau={self.tau})"
        return label

    def select_arm(self) -> int:
        scaled_values = self.values / self.tau
        # Estabilización numérica: restamos el máximo antes de aplicar exp
        max_scaled = np.max(scaled_values)
        exp_values = np.exp(scaled_values - max_scaled)
        probs = exp_values / np.sum(exp_values)
        return np.random.choice(self.k, p=probs)


    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        # Actualizamos la media incrementalmente (promedio incremental)
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        super().update(arm, reward)

    def reset(self):
        super().reset()
        self.values = np.zeros(self.k)
        self.counts = np.zeros(self.k)
