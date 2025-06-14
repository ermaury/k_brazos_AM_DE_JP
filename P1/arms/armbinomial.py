"""
Module: arms/armbernouilli.py
Description: Contains the implementation of the ArmBinomial class for the binomial distribution arm.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np
from arms import Arm 

class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializa un brazo con distribución binomial B(n, p).

        :param n: Número de intentos.
        :param p: Probabilidad de éxito en cada intento.
        """
        assert n > 0, "El número de intentos n debe ser mayor que 0."
        assert 0 <= p <= 1, "La probabilidad p debe estar entre 0 y 1."

        self.n = n
        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución binomial.

        :return: Número de éxitos en n intentos.
        """
        return np.random.binomial(self.n, self.p)

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución binomial.

        :return: Valor esperado de la distribución, que es n * p.
        """
        return self.n * self.p

    def __str__(self):
        """
        Representación en cadena del brazo binomial.

        :return: Descripción detallada del brazo binomial.
        """
        return f"ArmBinomial(n={self.n}, p={self.p:.3f})"

    @classmethod
    def generate_arms(cls, k: int, n_min: int = 5, n_max: int = 20):
        """
        Genera k brazos binomiales con parámetros aleatorios.

        :param k: Número de brazos a generar.
        :param n_min: Valor mínimo de intentos n.
        :param n_max: Valor máximo de intentos n.
        :return: Lista de instancias de ArmBinomial.
        """
        assert k > 0, "El número de brazos debe ser mayor que 0."
        assert n_min < n_max, "n_min debe ser menor que n_max."

        # Generar valores aleatorios de n y p
        n_values = np.random.randint(n_min, n_max + 1, size=k)
        p_values = np.random.rand(k)  # Probabilidades aleatorias entre 0 y 1

        arms = [ArmBinomial(n, p) for n, p in zip(n_values, p_values)]
        return arms



