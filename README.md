# The Astrological Estimation #
## Price-Volume Strength Oscillator ##

We define ${Y(t)}$ the **price-volume strength** oscillator as:

${Y(t) = \frac {(p_t-p_{t-1})}{p_t+p_{t-1}} \min(1,v_t/v_{t-1})}$, where ${p_t, p_{t-1}, v_t}$ and ${v_{t-1}}$.

This oscillator detect strong bullish (${Y(t) \rarr 1}$) or bearish (${Y(t) \rarr -1}$) behavior.

## Longitude-Motion Estimator for ${Y(t)}$ ##

We define ${L(t)}$ the **price-volume strength** estimator as a neural network whose input (namely ${X}$) is the table
shown below:

| ${A_1}$                  | ${B_1}$                 | ... | ${A_7}$               | ${B_7}$ 
|--------------------------|-------------------------|-----|-----------------------|--------------------------
| ${a_1 \cos(f_1λ_1(t_1))}$|${b_1 \cos(f_1λ_1(t_1))}$| ... |${a_7 \cos(f_7λ_7(t_1))}$|${b_7 \cos(f_7λ_7(t_1))}$
| ${a_1 \cos(f_1λ_1(t_2))}$|${b_1 \cos(f_1λ_1(t_2))}$| ... |${a_7 \cos(f_7λ_7(t_2))}$|${b_7 \cos(f_7λ_7(t_2))}$
| ...                    | ...                   | ... | ...                   | ... 
| ${a_1 \cos(f_1λ_1(t_n))}$|${b_1 \cos(f_1λ_1(t_n))}$| ... |${a_7 \cos(f_7λ_7(t_n))}$|${b_7 \cos(f_7λ_7(t_n))}$

Let's describes the values in the above table:

1. ${A_k}$ and ${B_k}$ are respectively **traction factor** and **motion factor** for planet ${k}$.
2. ${a_k}$ and ${b_k}$ are respectively **gravitationnal force** and **mean motion** of planet ${k}$.
3. ${f_k = \frac {2k\pi}{T}}$, where ${T}$ is the orbital period of planet ${k}$.
4. ${λ_k(t)}$ is the **heliocentric longitude** of planet ${k}$ at time ${t}$.

That table was designed assuming that W.D. Gann theories are statistically sound.

## Neural Network for ${Y(t)}$ ##

The naive design (current one) is as follows:
```python
def create_dnn_model(X_train_scaled):
    set_all_seeds()
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)), 
        tf.keras.layers.Dropout(0.1),     

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),  
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),  
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),  
 

        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model 
```

This design delivers mixed results included in the next table:

Stock| y predicted var | y test var | test loss | mae     
-----|-----------------|------------|-----------|---------
AAPL | 0.000053        | 0.000207   | 0.000314  | 0.013695
VZ   | 0.000485        | 0.000124   | 0.001036  | 0.026831


We have the challenge to improve this design to behave equally fine with a broader selection of stocks.
