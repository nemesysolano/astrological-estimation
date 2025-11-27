# Surfing with Market Micro-Cycles and Momemtum #

## Price-Volume Strength Oscillator ##

We define ${Y(t)}$ as the **price-volume strength** oscillator:

${Y(t) = \frac {(p_t-p_{t-1})}{p_t+p_{t-1}} \min(1,v_t/v_{t-1})}$, where

- ${p_t}$ price at time ${t}$
- ${p_{t-1}}$ price at time ${t-1}$
- ${v_t}$ traded volume at time ${t}$
- ${v_{t-1}}$ traded volume at time ${t-1}$

This oscillator detects strong bullish (${Y(t) \rarr 1}$) or bearish (${Y(t) \rarr -1}$) behavior. Moreover, ${p_t}$ represents ${h_t}$ (high price in OHLC bar) when
analysing resistance/bearish momentum, ${l_t}$ (low price in OHLC bar) when analysing support/bullish momentum or ${c_t}$ close price when analysing trends.

### Prive Volume Estimator for ${Y(t+1)}$ ###

We define ${\hat Y(t+1)}$ as the **price-volume strength** estimator for ${Y(t+1)}$.  ${\hat Y(t+1)}$ is a neural network whose feature will be selected from the listing table below.

Our first challenge is to findout the most performant subset from this table and the best model for predicting ${Y(t+1)}$. Later on we will introduce ${\hat Y(t+1)}$ in 
the features set for the breaking gap estimator (${\hat G(t+1)}$).

| Feature Name | Description | Notation |
| --- | --- | ---  |
| **Price-Time cos**        | In this document   | ${\cos(θ_1(t))}$ ... ${\cos(θ_4(t-1))}$ |
| **Price-Time sin**        | In this document   | ${\sin(θ_1(t))}$ ... ${\sin(θ_4(t-1))}$ |
| **Average True Range %** | Code: **market_module.py**/add_average_true_range_percentage | ${\mathbf{Atrp_{14}}(t-i)}$ |
| **Bollinger Band Width for Close** | Code: **market_module.py**/add_bollinger_bands_width | ${\mathbf{BBW_c}(t-i)}$ |
| **Bollinger Band Width for High** | Code: **market_module.py**/add_bollinger_bands_width | ${\mathbf{BBW_h}(t-i)}$ |
| **Bollinger Band Width for Low** | Code: **market_module.py**/add_bollinger_bands_width | ${\mathbf{BBW_l}(t-i)}$ |
| **Realized Volatility for Close** | Code: **market_module.py**/add_realized_volatility  | ${\mathbf{Rvo_c}(t-i)}$ |
| **Realized Volatility for High** | Code: **market_module.py**/add_realized_volatility  | ${\mathbf{Rvo_h}(t-i)}$ |
| **Realized Volatility for Close** | Code: **market_module.py**/add_realized_volatility  | ${\mathbf{Rvo_l}(t-i)}$ |
| **Relative Volatility Index** | Code: **market_module.py**/add_relative_volatility_index | ${\mathbf{RVI_{14,h}}(t-i)}$ |
| **Relative Volatility Index** | Code: **market_module.py**/add_relative_volatility_index | ${\mathbf{RVI_{14,l}}(t-i)}$ |
| **Relative Volume** | Code: **market_module.py**/add_relative_volume |  ${\mathbf{Rv}(t-i)}$ |
| **Fast Trend Run**  | In this document | ${R_{f}(t)}$ |
| **Slow Trend Run**  | In this document | ${R_{s}(t)}$ |
| **The Structural Direction**| In this Document | ${S_d(t)}$ 

The design of this input table is predicated on the statistical validity of W.D. Gann's theories.

## The Closest Extreme ##

These concepts refer to finding the nearest prior occurrence of a high or low price that is **structurally higher** or **lower** than the current price at ${t}$. The search is always backward in time.

### Closest Higher High (${h_↑(t)}$)

* **Definition:** The higher high closest to the current high price $h(t)$ in the **OHLC** bar at time ${t}$.
* **Formula:** Finds the $h(t-i)$ that is strictly greater than $h(t)$ while minimizing the lookback period ${i}$.

${h_↑(t) = h(t-i_{\max(h)})}$ where ${i_{{\max(h)}} = \min \{j <\in \mathbb{Z}^+> \mid h(t-j) > h(t)\}}$

### Closest Lower High (${h_↓(t)}$)

* **Definition:** The lower high closest to the current high price $h(t)$ in the **OHLC** bar at time ${t}$.
* **Formula:** Finds the $h(t-i)$ that is strictly less than $h(t)$ while minimizing the lookback period ${i}$.

${h_↓(t) = h(t-i_{\min(h)})}$ where ${i_{\min(h)} = \min \{j \in \mathbb{Z}^+ \mid h(t-j) < h(t)\}}$

### Closest Higher Low (${l_↑(t)}$)

* **Definition:** The higher low closest to the current low price $l(t)$ in the **OHLC** bar at time ${t}$.
* **Formula:** Finds the $l(t-i)$ that is strictly greater than $l(t)$ while minimizing the lookback period ${i}$.

${l_↑(t) = l(t-i_{\max(l)})}$ where ${i_{\max(l)} = \min \{j \in \mathbb{Z}^+ \mid l(t-j) > l(t)\}}$

### Closest Lower Low (${l_↓(t)}$)

* **Definition:** The lower low closest to the current low price $l(t)$ in the **OHLC** bar at time $t$.
* **Note:** The document uses the notation ${h_↓(t)}$ for this indicator. For consistency with the other low-price indicators, the more logical notation $l_↓(t)$ is used here, but the value is based on the comparison of low prices.
* **Formula:** Finds the $l(t-i)$ that is strictly less than $l(t)$ while minimizing the lookback period ${i}$.

${l_↓(t) = l(t-i_{\min(l)})}$ where ${i_{\min(l)} = \min \{j \in \mathbb{Z}^+ \mid l(t-j) < l(t)\}}$

## The Price-Time Angles ##

Consider these definitions

1. ${B(t) = \max(t-i_{\max(h)},\space t-i_{\min(h)},\space t-i_{\max(l)},\space t-i_{\max(l)})}$ , 
2. ${b(t) = \{\frac{t-i_{\max(h)}}{B(t)}, \space \frac{t-i_{\min(h)}}{B(t)}, \space \frac{t-i_{\max(l)}}{B(t)}, \space \frac{t-i_{\max(l)}}{B(t)}\}}$, 
3. ${C(t) = \max(h(t)-h_↑(t),\space h(t)-h_↓(t),\space l_↑(t)-l(t),\space l_↓(t)-l(t))}$ and
4. ${c(t) = \{\frac{h(t)-h_↑(t)}{C(t)}, \space \frac{h(t)-h_↓(t)}{C(t)}, \space \frac{l_↑(t)-l(t)}{C(t)}, \space \frac{l_↓(t)-l(t)}{C(t)}\}}$.

If we divide pairwise ${b_c(t)}$ by ${b_k(t)}$ and then apply ${\mathbf{arctan}}$ function to every element in the resulting list, we get four **price-time angles** ${θ_1(t)}$, ${θ_2(t)}$, ${θ_3(t)}$ and ${θ_4(t)}$ ruling t time ${t}$.


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

| ${Y_c(t-i)}$     | ${Y_h(t-i)}$     | ${Y_l(t-i)}$     | ${\hat Y_c(t-i)}$    | ${\hat Y_h(t-i)}$     | ${\hat Y_l(t-i)}$|${S_d(t-i)}$    |${\mathbf{Atrp_{14}}(t-i)}$    |
|------------------|------------------|------------------|----------------------|-----------------------|------------------|----------------|-------------------------------|
| ${Y_c(t)}$       | ${Y_h(t)}$       | ${Y_l(t)}$       | ${\hat Y_c(t)}$      | ${\hat Y_h(t)}$       | ${\hat Y_l(t)}$  |${S_d(t)}$      |${\mathbf{Atrp_{14}}(t)}$      |
| ...              | ...              | ...              | ...                  | ...                   | ...              |...             |...                            |
| ${Y_c(t-(n-1))}$ | ${Y_h(t-(n-1))}$ | ${Y_l(t-(n-1))}$ |${\hat Y_c(t-(n-1))}$ | ${\hat Y_h(t-(n-1))}$ | ${L_l(t-(n-1))}$ |${S_d(t-(n-1))}$|${\mathbf{Atrp_{14}}(t-(n-1))}$|

where ${i}$ ranges from ${0}$ to ${n-1}$. Additionally ${Y_c(t)}$, ${Y_h(t)}$, ${Y_l(t)}$ are ${Y(t)}$ values calculated for **close**, **high** and **low** prices respectively.

Our DNN predictor for ${G(t+1)}$ is denoted as ${\hat G(t+1)}$
