"""
Module: algorithms/__init__.py
Description: Contiene las importaciones y modulos/clases públicas del paquete arms.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

# Importación de módulos o clases
from .algorithm import Algorithm
from .epsilon_greedy import EpsilonGreedy
from .ucb1 import UCB1
from .ucb2 import UCB2
from .gradient_bandit import GradientBandit
from .softmax_bandit import SoftmaxBandit

# Lista de módulos o clases públicas
__all__ = ['Algorithm', 'EpsilonGreedy', 'UCB1', 'UCB2', 'GradientBandit', "SoftmaxBandit"]

