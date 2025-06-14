"""
Module: algorithms/gradient_bandit.py
Description: Implementación del algoritmo Gradiente de Preferencias
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

class GradientBandit(Algorithm):
    def __init__(self, k: int, alpha: float = 0.1, use_baseline: bool = True):
        """
        Inicializa el algoritmo de Gradiente de Preferencias.

        :param k: Número de brazos.
        :param alpha: Tasa de aprendizaje.
        :param use_baseline: Si se usa una media base para estabilizar el aprendizaje.
        """
        super().__init__(k)
        self.alpha = alpha
        self.use_baseline = use_baseline
        self.preferences = np.zeros(k)
        self.baseline = 0.0
        self.t = 0  # Contador de pasos

    def get_algorithm_label(self) -> str:
        label = type(self).__name__ + f" (alpha={self.alpha}, baseline={self.use_baseline})"
        return label

    def select_arm(self) -> int:
        exp_preferences = np.exp(self.preferences)
        probs = exp_preferences / np.sum(exp_preferences)
        return np.random.choice(self.k, p=probs)

    def update(self, arm: int, reward: float):
        exp_preferences = np.exp(self.preferences)
        probs = exp_preferences / np.sum(exp_preferences)

        if self.use_baseline:
            self.baseline += (reward - self.baseline) / (self.t + 1)
            baseline = self.baseline
        else:
            baseline = 0.0

        one_hot = np.eye(self.k)[arm]
        self.preferences += self.alpha * (reward - baseline) * (one_hot - probs)
        super().update(arm, reward)
        self.t += 1

    def reset(self):
        super().reset()
        self.preferences = np.zeros(self.k)
        self.baseline = 0.0
        self.t = 0
