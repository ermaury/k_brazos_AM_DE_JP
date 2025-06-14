"""
Module: algorithms/ucb1.py
Description: Implementación del algoritmo UCB1 para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np
from algorithms.algorithm import Algorithm

class UCB2(Algorithm):
    def __init__(self, k: int, alpha: float = 0.1):
        """
        Inicializa el algoritmo UCB2.

        :param k: Número de brazos.
        :param alpha: Parámetro de exploración en (0,1).
        """
        super().__init__(k)
        assert 0 < alpha < 1, "El parámetro alpha debe estar en el rango (0,1)."

        self.alpha = alpha
        self.k_a = np.zeros(k, dtype=int)  # Contador de épocas de cada brazo
        self.r = np.ones(k, dtype=int)
        self.t = 1  # Contador de pasos global

    def get_algorithm_label(self) -> str:
        """
        Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.
    
        :param algo: Instancia de un algoritmo.
        :type algo: Algorithm
        :return: Cadena descriptiva para el algoritmo.
        :rtype: str
        """
        label = type(self).__name__
        label += f" (alpha={self.alpha})"
        return label

  
    
    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB2.
        :return: Índice del brazo seleccionado.
        """
        total_counts = np.sum(self.counts)

        if total_counts < self.k:
            return total_counts  # Elegir cada brazo al menos una vez

        # Determinar si un brazo debe seguir siendo seleccionado
        for arm in range(self.k):
            if self.counts[arm] < self.r[arm]:
                return arm  # Continuar con el mismo brazo en su bloque de exploración

        # Calcular la ecuación de UCB2
        #confidence_bounds = self.values + np.sqrt((1 + self.alpha) * np.log((np.e * self.t) / (np.maximum(1, self.r)) + 1e-9) / (2 * (self.r + 1e-9)))
        confidence_bounds = self.values + np.sqrt((1 + self.alpha) * np.log1p((np.e * self.t / np.maximum(1, self.r)) - 1) / (2 * np.maximum(1, self.r)))


        # Seleccionar el brazo con mayor UCB2
        chosen_arm = np.argmax(confidence_bounds)

        # Calcular la duración del nuevo bloque de exploración para este brazo
        self.k_a[chosen_arm] += 1
        self.r[chosen_arm] = np.ceil((1 + self.alpha) ** self.k_a[chosen_arm]).astype(int)

        return chosen_arm

    def update(self, arm: int, reward: float):
        """
        Actualiza la información del brazo seleccionado.

        :param arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida al seleccionar ese brazo.
        """
        super().update(arm, reward)  # Usa la actualización de la clase padre
        self.t += 1  # Incrementar el tiempo global

    def reset(self):
        super().reset()
        self.k_a = np.zeros(self.k, dtype=int)  # Contador de épocas de cada brazo
        self.r = np.zeros(self.k, dtype=int)  # Bloque de exploración por brazo
        self.t = 1