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

class UCB1(Algorithm):
    def __init__(self, k: int, c: float = 1.0):
        """
        Inicializa el algoritmo UCB1 con un parámetro de ajuste c.

        :param k: Número de brazos.
        :param c: Parámetro de exploración (usualmente c=1).
        """
        super().__init__(k)
        self.c = c  # Parámetro de exploración

    def get_algorithm_label(self) -> str:
        """
        Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.
    
        :param algo: Instancia de un algoritmo.
        :type algo: Algorithm
        :return: Cadena descriptiva para el algoritmo.
        :rtype: str
        """
        label = type(self).__name__
        label += f" (c={self.c})"
        return label

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB1.
        :return: Índice del brazo seleccionado.
        """
        total_counts = np.sum(self.counts)

        if total_counts < self.k:
            return total_counts  # Seleccionar cada brazo al menos una vez

        confidence_bounds = self.values + self.c * np.sqrt(np.log(total_counts) / (self.counts + 1e-9))
        
        return np.argmax(confidence_bounds)