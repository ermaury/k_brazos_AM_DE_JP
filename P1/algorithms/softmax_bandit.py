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
        Inicializa el algoritmo Softmax (Boltzmann Exploration).

        :param k: Número de brazos.
        :param tau: Temperatura para suavizar la probabilidad.
        """
        super().__init__(k)
        assert tau > 0, "La temperatura tau debe ser positiva."
        self.tau = tau
        self.preferences = np.zeros(k)
        self.t = 0  # Contador de pasos
        self.baseline = 0.0

    def get_algorithm_label(self) -> str:
        label = type(self).__name__ + f" (tau={self.tau})"
        return label

    def select_arm(self) -> int:
        exp_preferences = np.exp(self.preferences / self.tau)
        probs = exp_preferences / np.sum(exp_preferences)
        return np.random.choice(self.k, p=probs)

    def update(self, arm: int, reward: float):
        exp_preferences = np.exp(self.preferences / self.tau)
        probs = exp_preferences / np.sum(exp_preferences)
        self.baseline += (reward - self.baseline) / (self.t + 1)
        baseline = self.baseline
        self.preferences += 0.1 * (reward - baseline) * (np.eye(self.k)[arm] - probs)
        self.t += 1
        super().update(arm, reward)

    def reset(self):
        super().reset()
        self.preferences = np.zeros(self.k)
        self.t = 0
        self.baseline = 0.0
