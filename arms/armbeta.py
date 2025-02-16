"""
Module: arms/armbernouilli.py
Description: Contains the implementation of the ArmBeta class for the beta distribution arm.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np
from arms import Arm  # Suponiendo que hay una clase base "Arm"

class ArmBeta(Arm):
    def __init__(self, alpha: float = 1, beta: float = 1):
        """
        Inicializa un brazo con distribución Beta Beta(α, β).

        :param alpha: Parámetro α de la distribución Beta (número de éxitos + 1).
        :param beta: Parámetro β de la distribución Beta (número de fracasos + 1).
        """
        assert alpha > 0 and beta > 0, "Los parámetros alpha y beta deben ser mayores a 0."
        self.alpha = alpha
        self.beta = beta

    def pull(self):
        """
        Genera una recompensa muestreando de la distribución Beta.

        :return: Un valor entre 0 y 1, muestreado de Beta(α, β).
        """
        return np.random.beta(self.alpha, self.beta)

    def update(self, reward: int):
        """
        Actualiza los parámetros α y β con la nueva observación.

        :param reward: 1 si el experimento tuvo éxito, 0 si fue un fracaso.
        """
        assert reward in [0, 1], "La recompensa debe ser 0 o 1 en un modelo Bernoulli."
        if reward == 1:
            self.alpha += 1
        else:
            self.beta += 1

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Beta.

        :return: Valor esperado de Beta(α, β), que es α / (α + β).
        """
        return self.alpha / (self.alpha + self.beta)

    def __str__(self):
        """
        Representación en cadena del brazo Beta.

        :return: Descripción detallada del brazo Beta.
        """
        return f"ArmBeta(α={self.alpha}, β={self.beta})"

    @classmethod
    def generate_arms(cls, k: int):
        """
        Genera k brazos Beta con parámetros iniciales α=1 y β=1 (priors uniformes).

        :param k: Número de brazos a generar.
        :return: Lista de instancias de ArmBeta.
        """
        assert k > 0, "El número de brazos debe ser mayor que 0."
        arms = [ArmBeta(1, 1) for _ in range(k)]
        return arms
