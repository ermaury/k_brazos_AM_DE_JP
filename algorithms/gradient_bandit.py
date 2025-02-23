"""
Module: algorithms/gradient_bandit.py
Description: Implementación del algoritmo Gradient Bandit (ascenso del gradiente) para el problema de los k-brazos.
Utiliza una política softmax basada en preferencias y actualiza dichas preferencias mediante un paso de gradiente,
utilizando como línea base el promedio incremental de recompensas.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.
For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np
from algorithms.algorithm import Algorithm

class GradientBandit(Algorithm):
    def __init__(self, k: int, alpha: float = 0.1):
        """
        Inicializa el algoritmo Gradient Bandit.

        :param k: Número de brazos.
        :param alpha: Tasa de aprendizaje para actualizar las preferencias.
        :raises AssertionError: Si alpha no es positiva.
        """
        assert alpha > 0, "El parámetro alpha debe ser mayor que 0."
        super().__init__(k)
        self.alpha = alpha
        self.H = np.zeros(k)            # Preferencias iniciales para cada brazo
        self.average_reward = 0.0         # Línea base: promedio incremental de recompensas
        self.t = 0                      # Contador de pasos
        self.probabilities = np.ones(k) / k  # Probabilidades iniciales (uniformes)

    def get_algorithm_label(self) -> str:
        """
        Genera una etiqueta descriptiva para el algoritmo.

        :return: Cadena descriptiva que incluye el parámetro alpha.
        """
        return f"GradientBandit (alpha={self.alpha})"

    def select_arm(self) -> int:
        """
        Calcula las probabilidades softmax a partir de las preferencias y selecciona un brazo.

        :return: Índice del brazo seleccionado.
        """
        # Para estabilidad numérica se resta el máximo de H
        expH = np.exp(self.H - np.max(self.H))
        self.probabilities = expH / np.sum(expH)
        # Seleccionar un brazo aleatoriamente de acuerdo a las probabilidades
        chosen_arm = np.random.choice(self.k, p=self.probabilities)
        return chosen_arm

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza la línea base (promedio de recompensa) y las preferencias usando la fórmula:
        
            H(a) = H(a) + alpha * (reward - baseline) * (I(a == chosen_arm) - pi(a))
        
        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida al tirar el brazo.
        """
        self.t += 1
        # Actualización incremental de la línea base (promedio)
        self.average_reward += (reward - self.average_reward) / self.t

        # Actualización de las preferencias para cada brazo
        for a in range(self.k):
            indicator = 1 if a == chosen_arm else 0
            self.H[a] += self.alpha * (reward - self.average_reward) * (indicator - self.probabilities[a])

    def reset(self):
        """
        Reinicia el estado del algoritmo.
        """
        self.counts = np.zeros(self.k, dtype=int)
        self.values = np.zeros(self.k, dtype=float)
        self.H = np.zeros(self.k)
        self.average_reward = 0.0
        self.t = 0
        self.probabilities = np.ones(self.k) / self.k
