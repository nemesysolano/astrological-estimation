# The Fourier-Astronomical Estimator #

## 🪐 Model Linking Planetary Positions to Asset Prices

We begin by defining the following astronomical and financial variables:

### Astronomical Variables
Let $N$ be the number of planets considered in the solar system.

* **$\Lambda(t) = \sum_{n=1}^{N} \lambda_{n}(t)$**: The **sum** of the **heliocentric longitudes** of the $N$ planets at time $t$.
* $m_n$: The **mass** of planet $n$.
* $d_n$: The **distance** between Earth and planet $n$, measured in **astronomical units (AU)**.

### Financial Variables
These variables are defined over a time period from $t_1$ to $t_n$.

* ${P(t)}$: The **price** of an asset (such as a stock, future, currency, or index fund) at time $t$.
* ${P_{\text{max}} = \max_{t_1...t_k} P(t)}$: The **maximum** asset price during the period.
* ${P_{\text{min}} = \min_{t_1...t_k} P(t)}$: The **minimum** asset price during the period.
* ${P_r = \frac{\sum_{t=t_1}^{t_k} P(t)}{k}}$: The **average reference price**.

---

## 📈 The Fourier-Astronomical Estimator $F(t)$

We define **$F(t)$** as the **Fourier estimator** for the asset price $P(t)$, which relates the financial data to the planetary influences via the following equation:

${F(t) \approx \frac{P(t)-P_r}{\sum_{n=1}^{N} \frac{m_n}{d_n^2}}}$

${F(t)}$ is modeled as a **Fourier series**, which provides the spectral decomposition of the relationship:

${F(t) = a_0 +\sum_{j=1}^{\infty}\left(a_j\cos\left(\boldsymbol{j\Lambda(t)}\right) + b_j\sin\left(\boldsymbol{j\Lambda(t)}\right)\right)}$


This series uses the **sum of the planets' heliocentric longitudes, $\Lambda(t)$**, as the **angular variable** (or phase) in its trigonometric terms.
