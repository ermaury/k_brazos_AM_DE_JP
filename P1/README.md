## 1️⃣ `bandit_experiment.ipynb` — Familia $\varepsilon$-Greedy

Contiene los experimentos realizados con el algoritmo **$\varepsilon$-greedy**, comparando tres niveles de exploración:

- $\varepsilon = 0.0$ (greedy puro)
- $\varepsilon = 0.01$ (exploración muy baja)
- $\varepsilon = 0.1$ (exploración moderada)

Se evalúa el rendimiento bajo tres tipos de distribuciones de recompensa:

- Normal
- Bernoulli
- Binomial

Para cada combinación se generan gráficos de recompensa media, selección óptima y regret acumulado. Además, al inicio del notebook se imprime el resumen numérico de las recompensas medias alcanzadas por cada configuración.

---

## 2️⃣ `bandit_experiment-UCB.ipynb` — Familia UCB (Upper Confidence Bound)

Contiene los experimentos realizados con las variantes **UCB1** y **UCB2**:

- **UCB1** con parámetros $c \in \{0.01,\ 1.0,\ 3.0\}$
- **UCB2** con parámetros $\alpha \in \{0.01,\ 0.3,\ 0.5\}$

Se aplican sobre las mismas tres distribuciones (Normal, Bernoulli y Binomial), con análisis completo de:

- Recompensas medias alcanzadas
- Porcentaje de selección del brazo óptimo
- Curvas de regret acumulado


---

## 3️⃣ `bandit_experiment_gradient.ipynb` — Familia Gradient (Softmax y Gradient Bandit)

Contiene los experimentos de los algoritmos basados en ascenso del gradiente:

- **Softmax Bandit** con temperaturas $\tau \in \{0.01,\ 0.5,\ 1.0\}$
- **Gradient Bandit** con tasas de aprendizaje $\alpha \in \{0.01,\ 0.05,\ 0.1\}$ (con baseline activado)

Se comparan nuevamente bajo distribuciones Normal, Bernoulli y Binomial. Se analiza el impacto de los hiperparámetros en:

- Recompensa media final
- Selección del brazo óptimo
- Evolución del regret

---

## Organización general

Cada notebook sigue la misma estructura:

1. Definición de entornos (brazos y distribuciones).
2. Definición de algoritmos.
3. Ejecución de múltiples repeticiones.
4. Cálculo de estadísticas finales.
5. Visualización de resultados gráficos.
6. Resumen numérico de recompensas medias por configuración.

