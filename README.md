# The Astrological Indicator #

## Price-Volume Strength Oscillator ##

We define ${Y(t)}$ as the **price-volume strength** oscillator:

${Y(t) = \frac {(p_t-p_{t-1})}{p_t+p_{t-1}} \min(1,v_t/v_{t-1})}$, where

- ${p_t}$ close price at time ${t}$
- ${p_{t-1}}$ close price at time ${t-1}$
- ${v_t}$ traded volume at time ${t}$
- ${v_{t-1}}$ traded volume at time ${t-1}$

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

# The Swing Ratio ${S(t)}$ #
Consider the bar sequence ${\mathbf{OHLC}}$ = ${(o_{t-n}, h_{t-n}, l_{t-n}, c_{t-n}),...,(o_t, h_t, l_t, c_t)}$ up to the current bar ${t}$; it's assumed that this sequence is longer than 2 bars. 

### Fast Trend Run ${R_{f}(t)}$ ##
The purpose of the **fast trend run** ${R_{f}(t)}$ is to quantify the magnitude of the last directional push. If the price breaks the structural support ${l_{t-3}}$
after only a small push (${R_{f}(t)}$ is small), then ${S(t)}$ will be large, indicating a severe, high-provability violation. 

Let ${a}$ and ${b}$ be two consecutive moments in the time series. A trend begins when either: ${c_a < c_b}$ or ${c_a > c_b}$. When a **fast trend** begins, we denote ${t_f = a}$ as the **starting point** of the **fast trend**, and:

- ${R_f(t) = c_t - c_{t_f}}$: The trend run.
- ${n = t - t_f}$: The duration of the trend run, measured as the number of bars excluding the start bar

### Slow Trend Run ${R_s(t)}$ ##

The purpose of the **slow trend run** is to gauge the magnitude of the run since the structural trend was first stablished. The formulae is quite similar except for the **starting point**:

- ${R_s(t) = c_t - c_{t_s-1}}$: The trend run.
- ${n = t - t_s}$: The duration of the trend run, measured as the number of bars excluding the start bar

Now we need a clear definition of ${t_s}$ and for that purpose we will introduce the **structural direction (${S_d(t)}$)** concept.

#### The Structural Direction ${S_d(t)}$ ####
The structural direction captures both structural support/resistance and a price low/high are moving in the same direction.

| Priority | Condition | Structural Trend | ${S_d(t)}$ |
|----------|-----------|------------------|------------|
| **1**    | ${h_t > h_{t-1}}$ and ${l_t \ge l_{t-1}}$ | **Ascending** (New High Structure) | +1 |
| **2**    | ${l_t < l_{t-1}}$ and ${h_t \le h_{t-1}}$ | **Descending** (New Low Structure) | -1 |
| **3**    | Otherwise (${h_t \le h_{t-1}}$ and ${l_t \ge l_{t-1}}$ **OR** ${h_t > h_{t-1}}$ and ${l_t < l_{t-1}}$) | **Neutral** (Continuation/Indecisive) | ${S_d(t-1)}$ |


The run only starts (${t_s=t}$) when the **structural direction changes**, signifying the beginning of a new run and the end of the previous one.

Scenario                                 |Condition              |Update for ${t_s}$
-----------------------------------------|-----------------------|----------------
Structural Reversal (Start of a new run) |${S_d(t) \ne S_d(t-1)}$| ${t_s = t}$
Trend Continuation (Run is Going)        |${S_d(t) = S_d(t-1)}$  | ${t_s = t_{s-1}}$

### The Breaking Gap ###

Suppose that a new bar ${(o_{t+1}, h_{t+1}, l_{t+1}, c_{t+1})}$ comes along violating the trend. If the trend is an ascending one, we define the breaking gap ${G(t+1)}$ the distance between 
the violated structural level and the extreme price that violated it. This is valuable for determining the age and total magnitude of the structural level being tested.

#### Trend Violation for Slow Ascending Trends ####

If the slow trend at time ${t}$ is **ascending**, the violation occurs when the price breaks **below the structural low** ${l_{t-1}}$.

If ${l_{t+1} < l_{t-1}}$, the Breaking Gap is defined as ${G(t+1) = l_{t-1} - l_{t+1}}$

#### Trend Violation for Slow Descending Trends ####

If the slow trend at time ${t}$ is **descending**, the violation occurs when the price breaks **above the structural high** ${h_{t-1}}$.

if ${h_{t+1} > h_{t-1}}$, the breaking Gap  is defined as ${G(t+1) = h_{t+1} - h_{t-1}}$

## The Fast Swing Ratio ${S_f(t)}$ ##
The **fast swing ratio** ${S_f(t)}$ is calculated using **Breaking Gap** ${G(t+1)}$ and the magnitude of the **fast trend** ${|R_f(t)|}$:

${S_f(t) = \mathbf{min}(2, (\frac {G(t+1)}{|R_f(t)|})^2)}$

## The Slow Swing Ratio ${S_s(t)}$ ##

${S_s(t) = \mathbf{min}(2, (\frac {R_s(t)}{|R^*_s(t)|})^2)}$.

Consider the ${R_s(t-n),...,R_s(t-1)}$ sequence of slow trend runs. We define ${R^*_s(t)}$ most recent element in that list whose direction is opposite to ${R_s(t)}$.
This means that progrfams must track ${R_s(t)}$ using a time series in order to find out ${R^*_s(t)}$.


# Directional Probabilities  #
The directional probabilities at time ${t}$ estimate the likelihood that the price will move in a particular direction in the next period ${t+1}$, as visualized in the Time-Price Plane (where the horizontal axis represents time and the vertical axis represents price). We assume that directional probabilities occur pair wise. When defining these probabilities, we have to deal with conflicting trends and aligned trends.

## Conflicting Trends ##

### Fast Ascending Trend vs Slow Descending Trend ###
If the breaking gap truncates an **fast ascending trend and the slow trend is descending**, then the directional probabilities are:

${P_↑(t) = \frac {S_f(t)}{S_f(t) + S_s(t)}}$ and 

${P_↓(t) = 1-P_↑(t)}$


### Fast Descending Trend vs Slow Ascending Trend ###
If the breaking gap truncates a **fast descending trend and the slow trend is ascending**, then the directional probabilities are:

${P_↑(t) = 1-P_↓(t)}$ and 

${P_↓(t) = \frac {S_f(t)}{S_f(t) + S_s(t)}}$

## Aligned Trends ##

In this case both threads are equals to ${P(t)}$ which is defined as

### Ascending Trends ###

When both trend are ascending

${P_↑(t) = \mathbf{min}(1,\frac {|S_s(t) + S_f(t)|} {2})}$

${P_↓ (t) = 1- P_↑(t)}$

### Descending Trends ###

${P_↑(t) = 1- P_↓(t)}$

${P_↓(t) = \mathbf{min}(1,\frac {|S_s(t) + S_f(t)|} {2})}$

# Estimating G(t+1) #
With ${G(t+1)}$ defined as the quantitative measure of the structural violation's magnitude, our next step is to engineer the DNN input features required for its prediction. The input table is

| ${Y(t_{j-i})}$ | ... | ${Y(t_{j-(i+k)})}$ | ${L(t_{j-i})}$ | ... | ${L(t_{j-(i+k)})}$ |
|----------------|-----|--------------------|----------------|-----|--------------------|
|${Y(t_{j-1})}$  | ... |${Y(t_{j-(1+k)})}$  |${L(t_{j-1})}$  | ... |${L(t_{j-(1+k)})}$  |
|${Y(t_{j-2})}$  | ... |${Y(t_{j-(2+k)})}$  |${L(t_{j-2})}$  | ... |${L(t_{j-(2+k)})}$  |
| ...            | ... | ...                | ...            | ... | ...                |
| ${Y(t_{j-n})}$ | ... |${Y(t_{j-(n+k)})}$  |${L(t_{j-n})}$  | ... |${L(t_{j-(n+k)})}$  |

