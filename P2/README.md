# Descripción de los notebooks de los experimentos de Aprendizaje por Refuerzo

Este repositorio contiene los experimentos realizados para comparar distintas familias de algoritmos de Aprendizaje por Refuerzo (Reinforcement Learning) tanto en entornos tabulares como continuos, evaluando múltiples aproximadores de función de valor.

---

## 1️⃣ `ExperimentoMetodosTabulares.ipynb` — Algoritmos Tabulares en FrozenLake

- **Entorno:** FrozenLake (problema discreto clásico de navegación estocástica).
- **Algoritmos evaluados:**
  - Monte Carlo (on-policy y off-policy)
  - SARSA (on-policy TD(0))
  - Q-Learning (off-policy TD(0))
- **Métricas comparadas:**
  - Tasa de éxito (recompensa media)
  - Longitud media de los episodios
  - Número de episodios hasta alcanzar ≥90% de éxito
  - Robustez frente a cambios de dinámica
- **Características:**
  - Implementación completamente tabular (tablas Q discretas).
  - Ajuste de hiperparámetros incluido para cada algoritmo.
  - Comparativa directa de estabilidad, velocidad de convergencia y sensibilidad a la estocasticidad.

---

## 2️⃣ `ExperimentoMetodosAproximados.ipynb` — Aproximación de Funciones en MountainCar

- **Entorno:** MountainCar-v0 (problema de control continuo con espacio de observación 2D continuo).
- **Algoritmos evaluados (SARSA semi-gradient + aproximadores):**
  - **Tile Coding** (baldosas superpuestas)
  - **RBF** (Radial Basis Functions gaussianas)
  - **Fourier Basis** (expansión armónica)
  - **DQN** (Deep Q-Learning con red neuronal)
- **Métricas comparadas:**
  - Recompensa acumulada por episodio
  - Evolución de la convergencia
  - Comparación de estabilidad entre los distintos métodos de aproximación
- **Características:**
  - Necesidad de aproximadores debido al espacio continuo de estados.
  - Estudio detallado de los parámetros clave de cada técnica (número de tilings, número de bases, orden de Fourier, tamaño de la red, etc.)
  - Evaluación directa de la capacidad de generalización de cada aproximador.

---
