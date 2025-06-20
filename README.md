# Extensiones de Machine Learning - Máster en Inteligencia Artificial (UMU)

## Información

- **Alumnos:** Marín Ortega, Antonio; Escudero de Paco, Darío; Pérez Pujalte, Francisco Javier
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2024/2025

## Descripción

Este trabajo tiene como objetivo el estudio práctico y comparativo de distintas familias de algoritmos de Aprendizaje por Refuerzo (Reinforcement Learning) y de problemas de Bandido de K-Brazos (K-Armed Bandit), analizando sus comportamientos bajo distintos entornos, estrategias de exploración y técnicas de aproximación.

Se han implementado y evaluado experimentalmente las principales variantes clásicas de ambos dominios, profundizando en los mecanismos de exploración-explotación, el efecto de los hiperparámetros, y la estabilidad de aprendizaje bajo diferentes condiciones de ruido y espacio de estados.

## Estructura

El repositorio se organiza en dos grandes bloques de experimentos:

### P1 - Bandido de K-Brazos (Práctica 1)

**Notebooks:**

- `bandit_experiment.ipynb`  
  Estudio del algoritmo $\varepsilon$-greedy bajo distintas configuraciones de exploración ($\varepsilon \in \{0.0,\ 0.01,\ 0.1\}$) sobre distribuciones Normal, Bernoulli y Binomial.

- `bandit_experiment-UCB.ipynb`  
  Evaluación de los algoritmos UCB1 y UCB2 con múltiples parámetros (UCB1: $c \in \{0.01,\ 1.0,\ 3.0\}$; UCB2: $\alpha \in \{0.01,\ 0.3,\ 0.5\}$).

- `bandit_experiment_gradient.ipynb`  
  Comparativa de algoritmos de gradiente de políticas: Softmax (con temperaturas $\tau \in \{0.01,\ 0.5,\ 1.0\}$) y Gradient Bandit (con tasas de aprendizaje $\alpha \in \{0.01,\ 0.05,\ 0.1\}$), evaluados sobre las mismas distribuciones.

Cada notebook incluye:

- Definición del entorno y algoritmos.
- Entrenamiento multi-repetición.
- Evaluación de recompensas medias, selección óptima y regret.
- Resúmenes numéricos imprimibles y gráficos comparativos.

---

### P2 - Aprendizaje por Refuerzo (Práctica 2)

**Notebooks:**

- `ExperimentoMetodosTabulares.ipynb`  
  Evaluación de algoritmos tabulares sobre el entorno FrozenLake (discreto):

  - Monte Carlo (on-policy y off-policy)
  - SARSA (TD(0), on-policy)
  - Q-Learning (TD(0), off-policy)

  Incluye análisis de tasa de éxito, longitud de episodios, episodios hasta convergencia y robustez ante cambios de dinámica.

- `ExperimentoMetodosAproximados.ipynb`  
  Evaluación de métodos de control aproximado sobre el entorno continuo MountainCar:

  - SARSA semi-gradiente con Tile Coding
  - SARSA semi-gradiente con RBF
  - SARSA semi-gradiente con Fourier Basis
  - DQN (Deep Q-Learning)

  Incluye análisis comparativo de velocidad de convergencia, estabilidad, y capacidad de generalización de cada aproximador de función de valor.

---

## Instalación y Uso

- Requiere instalación de Python 3.10+ y las siguientes librerías:

  - `numpy`
  - `matplotlib`
  - `gymnasium`
  - `tqdm`
  - `torch` (para DQN)

- Para ejecutar los notebooks:

```bash
pip install -r requirements.txt
