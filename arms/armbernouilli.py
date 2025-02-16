"""
Module: arms/armbernouilli.py
Description: Contains the implementation of the ArmBernouilli class for the bernouilli distribution arm.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np
from arms import Arm

class ArmBernouilli(Arm):
    def __init__(self, p: float):
        """
        Inicializa un brazo Bernoulli con probabilidad de éxito p.

        :param p: Probabilidad de obtener recompensa 1 (éxito).
        """
        assert 0 <= p <= 1, "La probabilidad p debe estar entre 0 y 1."
        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución Bernoulli.

        :return: 1 con probabilidad p, 0 con probabilidad (1 - p).
        """
        return np.random.rand() < self.p  # Retorna 1 con probabilidad p, 0 con 1-p

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Bernoulli.

        :return: p (ya que E[X] = p en Bernoulli).
        """
        return self.p

    def __str__(self):
        """
        Representación en cadena del brazo Bernoulli.

        :return: Descripción detallada del brazo Bernoulli.
        """
        return f"ArmBernouilli(p={self.p})"

    @classmethod
    def generate_arms(cls, k: int):
        """
        Genera k brazos Bernoulli con probabilidades únicas.

        :param k: Número de brazos a generar.
        :return: Lista de instancias de BernoulliArm.
        """
        assert k > 0, "El número de brazos debe ser mayor a 0."

        p_values = np.random.rand(k)
        arms = [ArmBernouilli(p) for p in p_values]

        return arms


