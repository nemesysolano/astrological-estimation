# The Astrological Estimation #

## Price-Volume Strength Oscillator ##

We define ${Y(t)}$ as the **price-volume strength** oscillator:

${Y(t) = \frac {(p_t-p_{t-1})}{p_t+p_{t-1}} \min(1,v_t/v_{t-1})}$, where ${p_t, p_{t-1}, v_t}$ and ${v_{t-1}}$.

This oscillator detects strong bullish (${Y(t) \rarr 1}$) or bearish (${Y(t) \rarr -1}$) behavior.

## Longitude-Motion Estimator for ${Y(t)}$ ##

We define ${L(t)}$ as the **price-volume strength** estimator for ${Y(t)}$. ${L(t)}$ is a neural network whose input is the table ${Atr_{14}(t_{j-1})}$           
presented below:

| ${A_1}$                     | ${B_1}$                     | ... | ${A_7}$                     | ${B_7}$                     | ${Y(t_{j-i})}$ | ... | ${Y(t_{j-(i+3)})}$ | ${\mathbf{Atr_{14}(t_{j-i})}}$ | ${\mathbf{Rv(t_{j-i})}}$ |
|-----------------------------|-----------------------------|-----|-----------------------------|-----------------------------|----------------|-----|--------------------|--------------------------------|--------------------------|
|${a_1 \cos(f_1λ_1(t_{j-1}))}$|${b_1 \cos(f_1λ_1(t_{j-1}))}$| ... |${a_7 \cos(f_7λ_7(t_{j-1}))}$|${b_7 \cos(f_7λ_7(t_{j-1}))}$|${Y(t_{j-1})}$  | ... |${Y(t_{j-4})}$      | ${Atr_{14}(t_{j-1})}$          | ${\mathbf{Rv(t_{j-1})}}$ |
|${a_1 \cos(f_1λ_1(t_{j-2}))}$|${b_1 \cos(f_1λ_1(t_{j-2}))}$| ... |${a_7 \cos(f_7λ_7(t_{j-2}))}$|${b_7 \cos(f_7λ_7(t_{j-2}))}$|${Y(t_{j-2})}$  | ... |${Y(t_{j-5})}$      | ${Atr_{14}(t_{j-2})}$          | ${\mathbf{Rv(t_{j-2})}}$ |
| ...                         | ...                         | ... | ...                         | ...                         | ...            | ... | ...                | ...                            | ...                      |
|${a_1 \cos(f_1λ_1(t_{j-n}))}$|${b_1 \cos(f_1λ_1(t_{j-n}))}$| ... |${a_7 \cos(f_7λ_7(t_{j-n}))}$|${b_7 \cos(f_7λ_7(t_{j-n}))}$|${Y(t_{j-n})}$  | ... |${Y(t_{j-(n+3)})}$  | ${Atr_{14}(t_{j-n})}$          | ${\mathbf{Rv(t_{j-n})}}$ |


where:

The values in the above table are described as follows:

1. ${A_k}$ and ${B_k}$ are the **traction** and **motion** factors, respectively, for planet ${k}$.
2. ${a_k}$ and ${b_k}$ are the **gravitational** and **motion** factors, respectively, for planet ${k}$.
3. ${f_k = \frac {2k\pi}{T}}$, where ${T}$ denotes the orbital period of planet ${k}$.
4. ${λ_k(t_{j-i})}$ represents the **heliocentric longitude** of planet ${k}$ at time ${t_{j-i}}$.
5. ${\mathbf{Atr_{14}(t_{j-i})}}$ signifies the 14-day **average true range** (implying that ${t > 14}$ days).
6. ${Y(t_{j-i})}$ to ${Y(t_{j-(i+3)})}$ are the values for the four most recent days, starting from and including ${t_{j-i}}$.
7. ${\mathbf{Rv(t_{j-i})}}$ is ${\frac{v_{t-i}}{o_{t-i}}}$ signifies the lagged **Relative Volume** (or Volume Ratio), defined as the ratio between traded volumeß (${v_{t-i}}$) and outstanding shares (${o_{t-i}}$).

The design of this input table is predicated on the statistical validity of W.D. Gann's theories.
