# The Astrological Estimation #
## Price-Volume Strength Oscillator ##

We define ${Y(t)}$ the **price-volume strength** oscillator as:

${Y(t) = \frac {(p_t-p_{t-1})}{p_t+p_{t-1}} \min(1,v_t/v_{t-1})}$, where ${p_t, p_{t-1}, v_t}$ and ${v_{t-1}}$.

This oscillator detect strong bullish (${Y(t) \rarr 1}$) or bearish (${Y(t) \rarr -1}$) behavior.

## Longitude-Motion Estimator for ${Y(t)}$ ##

We define ${L(t)}$ the **price-volume strength** estimator as a neural network whose input (namely ${X}$) is the table
shown below:

| ${A_1}$                     | ${B_1}$                     | ... | ${A_7}$                     | ${B_7}$                               | ${\text{ATR}_{14}(t)}$         | ${Y(t_{i-1})}$   | ${Y(t_{i-2})}$
|-----------------------------|---------------------------- |-----|-----------------------------|-----------------------------|------------------------------------------|------------------|------------------
|${a_1 \cos(f_1λ_1(t_j))}$    |${b_1 \cos(f_1λ_1(t_j))}$    | ... |${a_7 \cos(f_7λ_7(t_j))}$    |${b_7 \cos(f_7λ_7(t_j))}$    |${\text{ATR}_{14}(t_j)}$                  |${Y(t_{j-1})}$    |${Y(t_{j-2})}$
|${a_1 \cos(f_1λ_1(t_{j+1}))}$|${b_1 \cos(f_1λ_1(t_{j+1}))}$| ... |${a_7 \cos(f_7λ_7(t_{j+1}))}$|${b_7 \cos(f_7λ_7(t_{j+1}))}$|${\text{ATR}_{14}(t_{j+1})}$              |${Y(t_{{j+1}-1})}$|${Y(t_{{j+1}-2})}$
| ...                         | ...                         | ... | ...                         | ...                         |                                          | ...              | ...             
|${a_1 \cos(f_1λ_1(t_{j+n}))}$|${b_1 \cos(f_1λ_1(t_{j+n}))}$| ... |${a_7 \cos(f_7λ_7(t_{j+n}))}$|${b_7 \cos(f_7λ_7(t_{j+n}))}$|${\text{ATR}_{14}(t_{j+n})}$              |${Y(t_{{j+n}-1})}$|${Y(t_{{j+n}-2})}$
where

Let's describes the values in the above table:

1. ${A_k}$ and ${B_k}$ are respectively **traction factor** and **motion factor** for planet ${k}$.
2. ${a_k}$ and ${b_k}$ are respectively **gravitationnal force** and **mean motion** of planet ${k}$.
3. ${f_k = \frac {2k\pi}{T}}$, where ${T}$ is the orbital period of planet ${k}$.
4. ${λ_k(t)}$ is the **heliocentric longitude** of planet ${k}$ at time ${t}$.

That table was designed assuming that W.D. Gann theories are statistically sound.
