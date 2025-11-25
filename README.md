# The Astrological Indicator #

## Price-Volume Strength Oscillator ##

We define ${Y(t)}$ as the **price-volume strength** oscillator:

${Y(t) = \frac {(p_t-p_{t-1})}{p_t+p_{t-1}} \min(1,v_t/v_{t-1})}$, where

- ${p_t}$ price at time ${t}$
- ${p_{t-1}}$ price at time ${t-1}$
- ${v_t}$ traded volume at time ${t}$
- ${v_{t-1}}$ traded volume at time ${t-1}$

This oscillator detects strong bullish (${Y(t) \rarr 1}$) or bearish (${Y(t) \rarr -1}$) behavior. Moreover, ${p_t}$ represents ${h_t}$ (high price in OHLC bar) when
analysing resistance/bearish momentum, ${l_t}$ (low price in OHLC bar) when analysing support/bullish momentum or ${c_t}$ close price when analysing trends.

## Longitude-Motion Estimator for ${Y(t)}$ ##

We define ${L(t)}$ as the **price-volume strength** estimator for ${Y(t)}$. ${L(t)}$ is a neural network whose input is illustrated by the table below:

| Feature Name | Feature Type | Price Series | Notation | Strategic Role |
| :--- | :--- | :--- | :--- | :--- |
|  ✔ **Astrological Longitude Cosine** | Astrological | N/A | ${a_1 \cos(f_1λ_1(t))}$ ... ${a_7 \cos(f_1λ_1(t))}$ | Measures the sine component of the angular separation for the 7 primary moving bodies. |
|  ✔ **Astrological Longitude Sine** | Astrological | N/A |${a_1 \sin(f_1λ_1(t))}$ ... ${a_7 \sin(f_1λ_1(t))}$ | Measures the cosine component of the angular separation for the 7 primary moving bodies. |
|  ✔ **Price-Volume Strength Osc.** | Price-Volume | Close ($\mathbf{c_t}$) | ${\mathbf{Y_c}(t-i)}$ | Price-Volume strength on the Close price series. |
|  ✔ **Price-Volume Strength Osc.** | Price-Volume | High ($\mathbf{h_t}$) | ${\mathbf{Y_h}(t-i)}$ | Price-Volume strength on the High price series. |
|  ✔ **Price-Volume Strength Osc.** | Price-Volume | Low ($\mathbf{l_t}$) | ${\mathbf{Y_l}(t-i)}$ | Price-Volume strength on the Low price series. |
|  ✔ **Average True Range %** | Volatility (Magnitude) | General (TR-based) | ${\mathbf{Atrp_{14}}(t-i)}$ | Overall volatility baseline for the model. |
|  ✔ **Bollinger Band Width** | Volatility (Cycle) | Close ($\mathbf{c_t}$) | ${\mathbf{BBW_c}(t-i)}$ | Measures general volatility cycle over the closing price. |
|  ✔ **Bollinger Band Width** | Volatility (Cycle) | High ($\mathbf{h_t}$) | ${\mathbf{BBW_h}(t-i)}$ | Volatility cycle specifically near **Resistance** (upper boundary). |
|  ✔ **Bollinger Band Width** | Volatility (Cycle) | Low ($\mathbf{l_t}$) | ${\mathbf{BBW_l}(t-i)}$ | Volatility cycle specifically near **Support** (lower boundary). |
|  ✔ **Realized Volatility** | Volatility (Magnitude) | Close ($\mathbf{c_t}$) | ${\mathbf{Rvo_c}(t-i)}$ | General magnitude signal of price movement. |
|  ✔ **Realized Volatility** | Volatility (Magnitude) | High ($\mathbf{h_t}$) | ${\mathbf{Rvo_h}(t-i)}$ | **Magnitude** of movement specifically near $\uparrow$ resistance. |
|  ✔ **Realized Volatility** | Volatility (Magnitude) | Low ($\mathbf{l_t}$) | ${\mathbf{Rvo_l}(t-i)}$ | **Magnitude** of movement specifically near $\downarrow$ support. |
|  ✔ **Relative Volatility Index** | Volatility (Direction) | High ($\mathbf{h_t}$) | ${\mathbf{RVI_{14,h}}(t-i)}$ | **Directional Bias** (Up/Down strength) near $\uparrow$ resistance. |
|  ✔ **Relative Volatility Index** | Volatility (Direction) | Low ($\mathbf{l_t}$) | ${\mathbf{RVI_{14,l}}(t-i)}$ | **Directional Bias** (Up/Down strength) near $\downarrow$ support. |
|  ✔ **Relative Volume** | Price-Volume | General (Volume) | ${\mathbf{Rv}(t-i)}$ | Volume strength baseline, independent of price series. |


The design of this input table is predicated on the statistical validity of W.D. Gann's theories.

# The Swing Ratio ${S(t)}$ #
Consider the bar sequence ${\mathbf{OHLC}}$ = ${(o_{t-n}, h_{t-n}, l_{t-n}, c_{t-n}),...,(o_t, h_t, l_t, c_t)}$ up to the current bar ${t}$; it's assumed that this sequence is longer than 2 bars. 

### Fast Trend Run ${R_{f}(t)}$ ##
The purpose of the **fast trend run** ${R_{f}(t)}$ is to quantify the magnitude of the last directional push. If the price breaks the structural support ${l_{t-3}}$
after only a small push (${R_{f}(t)}$ is small), then ${S(t)}$ will be large, indicating a severe, high-probability violation. 

Let ${a}$ and ${b}$ be two consecutive moments in the time series. A trend begins when either: ${c_a < c_b}$ or ${c_a > c_b}$. When a **fast trend** begins, we denote ${t_f = a}$ as the **starting point** of the **fast trend**, and:

- ${R_f(t) = c_t - c_{t_f}}$: The trend run.
- ${n = t - t_f}$: The duration of the trend run, measured as the number of bars excluding the start bar

### Slow Trend Run ${R_s(t)}$ ##

The purpose of the **slow trend run** is to gauge the magnitude of the run since the structural trend was first established. The formulae is quite similar except for the **starting point**:

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

----
If neither ${l_{t+1} < l_{t-1}}$ or ${h_{t+1} > h_{t-1}}$ occurs, then ${G(t+1)}$ is zero regardless trends nature.

## The Fast Swing Ratio ${S_f(t)}$ ##
The **fast swing ratio** ${S_f(t)}$ is calculated using **Breaking Gap** ${G(t+1)}$ and the magnitude of the **fast trend** ${|R_f(t)|}$:

${S_f(t) = \mathbf{min}(2, (\frac {G(t+1)}{|R_f(t)|})^2)}$

## The Slow Swing Ratio ${S_s(t)}$ ##

${S_s(t) = \mathbf{min}(2, (\frac {R_s(t)}{|R^*_s(t)|})^2)}$.

Consider the ${R_s(t-n),...,R_s(t-1)}$ sequence of slow trend runs. We define the **last opposite$ to ${R_s(t)}$** (namely ${R^*_s(t)}$) as the most recent element in that list whose direction is opposite to ${R_s(t)}$.
This means that programs must track ${R_s(t)}$ using a time series in order to find out ${R^*_s(t)}$.


# Directional Probabilities  #
The directional probabilities at time ${t+1}$ estimate the likelihood that the price will move in a particular direction in the next period (${t+1}$), as visualized in the Time-Price Plane (where the horizontal axis represents time and the vertical axis represents price). We assume that directional probabilities occur pair wise. When defining these probabilities, we have to deal with conflicting trends and aligned trends.

## Conflicting Trends ##
In this case slow and fast trends run in the opposite directions, the probabilities are calculated as follows:

### Fast Ascending Trend vs Slow Descending Trend ###
If the breaking gap truncates a **fast ascending trend and the slow trend is descending**, then the directional probabilities are:

${P_↑(t+1) = 1 - P_↓(t+1)}$, and

${P_↓(t+1) = \frac {S_f(t)}{S_f(t) + S_s(t)}}$ when ${S_f(t) \gt 0}$ or 0.5 otherwise.


### Fast Descending Trend vs Slow Ascending Trend ###
If the breaking gap truncates a **fast descending trend and the slow trend is ascending**, then the directional probabilities are:

${P_↑(t+1) = \frac {S_f(t)}{S_f(t) + S_s(t)}}$ when ${S_f(t) \gt 0}$ or 0.5 otherwise. And, 

${P_↓(t+1) = 1- P_↑(t+1)}$
 

## Aligned Trends ##

In this case both trends run in the same direction, the probabilities are calculated as follows:

### Ascending Trends ###

When both trend are ascending

${P_↑(t+1) = \mathbf{min}(1,\frac {|S_s(t) + S_f(t)|} {2})}$

${P_↓(t+1) = 1- P_↑(t+1)}$

### Descending Trends ###

${P_↑(t+1) = 1- P_↓(t+1)}$

${P_↓(t+1) = \mathbf{min}(1,\frac {|S_s(t) + S_f(t)|} {2})}$

# Estimating G(t+1) #
With ${G(t+1)}$ defined as the quantitative measure of the structural violation's magnitude, our next step is to engineer the DNN input features required for its prediction. The input table is

| ${Y_c(t-i)}$     | ${Y_h(t-i)}$     | ${Y_l(t-i)}$     | ${L_c(t-i)}$     | ${L_h(t-i)}$     | ${L_l(t-i)}$     |${S_d(t-i)}$    |${\mathbf{Atrp_{14}}(t-i)}$    |
|------------------|------------------|------------------|------------------|------------------|------------------|----------------|-------------------------------|
| ${Y_c(t)}$       | ${Y_h(t)}$       | ${Y_l(t)}$       | ${L_c(t)}$       | ${L_h(t)}$       | ${L_l(t)}$       |${S_d(t)}$      |${\mathbf{Atrp_{14}}(t)}$      |
| ...              | ...              | ...              | ...              | ...              | ...              |...             |...                            |
| ${Y_c(t-(n-1))}$ | ${Y_h(t-(n-1))}$ | ${Y_l(t-(n-1))}$ | ${L_c(t-(n-1))}$ | ${L_h(t-(n-1))}$ | ${L_l(t-(n-1))}$ |${S_d(t-(n-1))}$|${\mathbf{Atrp_{14}}(t-(n-1))}$|

where ${i}$ ranges from ${0}$ to ${n-1}$. Additionally ${Y_c(t)}$, ${Y_h(t)}$, ${Y_l(t)}$ are ${Y(t)}$ values calculated for **close**, **high** and **low** prices respectively.

Our DNN predictor for ${G(t+1)}$ is denoted as ${\hat G(t+1)}$
