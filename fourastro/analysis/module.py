import os
import random
from market import load_market_data
import pandas as pd 
import astro
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from itertools import combinations
Dense, Dropout, BatchNormalization, Concatenate, Input = (tf.keras.layers.Dense, tf.keras.layers.Dropout, tf.keras.layers.BatchNormalization, tf.keras.layers.Concatenate, tf.keras.layers.Input)
Model = tf.keras.models.Model
AdamW = tf.keras.optimizers.AdamW
import numpy as np

LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
l2 = tf.keras.regularizers.l2

planet_name_pairs = combinations([planet[1] for planet in astro.planets], 2  )

def clean_nan_and_inf(dataset):
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset.dropna(inplace=True)
    return dataset

def pct_difference(a, b):
    return 2*(b - a) / (a+b)

def atr0(atr, t):
    val = None
    if t in atr.index:
        val = atr[t]
    else:
        val = atr[atr.index > t].iloc[0]
    return val        
    
def define_X(Y_combined, atr, relative_volume): 
    astro_constants = astro.astro_constants
    X_astro = []
    print(Y_combined)
    # Start columns with the two lagged features
    columns = ['ψ_1', 'ψ_2', 'ATR','Rv'] 
    
    # Append the names for the astrological features
    columns.extend([ f"A_{i}" for i in range(len(astro.planets) * 2)])
    for t in Y_combined.index: 
        x = [Y_combined.loc[t, 'ψ_1'], Y_combined.loc[t, 'ψ_2'], atr[t], relative_volume[t]]
        k = 1
        for planet in astro.planets:
            # Astrological calculation remains the same, calculating for time t
            planet_name = planet[1]
            λ = astro_constants[planet_name]['λ'][t]
            a = astro_constants[planet_name]['g']
            b = astro_constants[planet_name]['b']
            T = astro_constants[planet_name]['T']
            f = 2 * k * np.pi /T            
            x.append(a * np.cos(f * λ))
            x.append(b * np.sin(f * λ))
            k+=1
        X_astro.append(x)        

    X = pd.DataFrame(X_astro, index=Y_combined.index, columns=columns)
    return X


def define_Y(dataset, column_name):
    # Causal Calculation of Y(t) = [(p_t - p_{t-1}) / (p_t + p_{t-1})] * min(1, v_t / v_{t-1})
    
    # 1. Align data and calculate components
    s_index = dataset.index[1:].copy() 
    current_price = dataset[column_name].iloc[1:].values
    previous_price = dataset[column_name].iloc[:-1].values 
    current_volume = dataset['Volume'].iloc[1:].values
    previous_volume = dataset['Volume'].iloc[:-1].values

    price_pct_diff = 2 * (current_price - previous_price) / (current_price + previous_price)
    volume_exp_diff = np.minimum(1, current_volume / previous_volume) 
    
    # 2. Create Y(t) series
    Y = pd.DataFrame(price_pct_diff * volume_exp_diff, index=s_index, columns=['ψ'])
    
    # 3. Create lagged features
    Y_lag1 = Y['ψ'].shift(periods=1)
    Y_lag1.name = 'ψ_1'
    
    Y_lag2 = Y['ψ'].shift(periods=2) 
    Y_lag2.name = 'ψ_2'
    
    # 4. Combine and clean
    Y_combined = pd.concat([Y['ψ'], Y_lag1, Y_lag2], axis=1)
    Y_combined.dropna(inplace=True)
    
    return Y_combined

def define_variables(train_data, validation_data, test_data, column_name):    
    Y_train_combined = define_Y(train_data, column_name)
    Y_val_combined = define_Y(validation_data, column_name)
    Y_test_combined = define_Y(test_data, column_name)

    Y_train_unscaled = Y_train_combined['ψ']
    Y_val_unscaled = Y_val_combined['ψ']
    Y_test_unscaled =Y_test_combined['ψ']

    X_train_unscaled = define_X(Y_train_combined, train_data['ATR'], train_data['relative_volume'])
    X_val_unscaled = define_X(Y_val_combined, validation_data['ATR'], validation_data['relative_volume'])
    X_test_unscaled = define_X(Y_test_combined, test_data['ATR'], test_data['relative_volume'])

    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train_unscaled)
    X_val_scaled = X_scaler.transform(X_val_unscaled)
    X_test_scaled = X_scaler.transform(X_test_unscaled)

    Y_scaler =  MinMaxScaler( feature_range=(-1, 1))
    Y_train_scaled = Y_train_unscaled.to_frame()
    Y_val_scaled = Y_val_unscaled.to_frame()
    Y_test_scaled = Y_test_unscaled.to_frame()
    
    
    return (
        X_train_scaled, X_val_scaled, X_test_scaled,
        Y_train_scaled, Y_val_scaled, Y_test_scaled,
        X_scaler, Y_scaler
    )

def set_all_seeds(seed_value=42):
    """Sets seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    # Python randomness
    random.seed(seed_value)
    # NumPy randomness
    np.random.seed(seed_value)
    # TensorFlow/Keras randomness
    tf.random.set_seed(seed_value)
    
def refined_dnn_model(X_train_scaled):
    set_all_seeds()
    input_dim = X_train_scaled.shape[1] 
    regularizer = l2(1e-4) # Define a small L2 penalty
    
    # ------------------ Input Layer ------------------
    input_tensor = Input(shape=(input_dim,))

    # Indices: 0-3 for Market Context (ψ_1, ψ_2, ATR, Rv)
    market_input = input_tensor[:, 0:4] 
    # Indices: 4 to End for Astrological Features
    astro_input = input_tensor[:, 4:]     

    # ------------------ Branch 1: Market Context (Strong Signal) ------------------
    # ADDED L2 REGULARIZATION
    market_branch = Dense(16, activation='relu', kernel_regularizer=regularizer, name='market_feature_proc')(market_input)
    market_branch = BatchNormalization()(market_branch)
    
    # ------------------ Branch 2: Astrological Features (Weak/Complex Signal) ------------------
    # ADDED L2 REGULARIZATION
    astro_branch = Dense(32, activation='relu', kernel_regularizer=regularizer, name='astro_feature_proc')(astro_input)
    astro_branch = BatchNormalization()(astro_branch)
    
    # ------------------ Merge and Deep Processing ------------------
    # Concatenate the processed features
    merged = Concatenate()([market_branch, astro_branch]) 
    
    # Deep layers for cross-feature interaction
    # ADDED L2 REGULARIZATION
    x = Dense(64, activation='relu', kernel_regularizer=regularizer)(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x) 
    
    # ADDED L2 REGULARIZATION
    x = Dense(32, activation='relu', kernel_regularizer=regularizer)(x)
    x = Dropout(0.1)(x)
    
    # Output layer (tanh forces output to [-1, 1] range)
    output_tensor = Dense(1, activation='tanh')(x)
    
    # Define the full model
    model = Model(inputs=input_tensor, outputs=output_tensor)
    
    # Use AdamW optimizer for better regularization
    model.compile(optimizer=AdamW(learning_rate=0.001), 
                  loss='mse', 
                  metrics=['mae'])
    
    return model

    
def add_atr_to_dataframe(data, window=14):
    high_low = data['High'] - data['Low']
    high_close_prev = np.abs(data['High'] - data['Close'].shift(1))
    low_close_prev = np.abs(data['Low'] - data['Close'].shift(1))

    true_range = pd.DataFrame({
        'high_low': high_low,
        'high_close_prev': high_close_prev,
        'low_close_prev': low_close_prev
    }).max(axis=1)

    true_range = true_range.shift(1)
    data['ATR'] = true_range.ewm(span=window, adjust=False).mean()
    data.dropna(inplace=True)
    return data
   
def lag_relative_volume(data, window=1):
    data['relative_volume'] = data['relative_volume'].shift(window)
    data.dropna(inplace=True)
    return data
    
def forecast(ticker):
    data = lag_relative_volume(add_atr_to_dataframe(clean_nan_and_inf(load_market_data(ticker))))
    # Split index into 70% 20% 10% respectively for train, validate and test
    data_index = data.index
    total_len = len(data_index)
    train_len = int(0.7 * total_len)
    val_len = int(0.2 * total_len)
    
    train_index = data_index[:train_len]
    val_index = data_index[train_len:train_len + val_len]
    test_index = data_index[train_len + val_len:]

    # Create train, validate and test data sets
    train_data = data.loc[train_index]
    validation_data = data.loc[val_index]
    test_data = data.loc[test_index]
    
    X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, _, _ = define_variables(train_data, validation_data, test_data, 'Close')
    
    # Define ModelCheckpoint callback to save the best model
    checkpoint_filepath =  os.path.join(os.path.dirname(__file__), 'models', f"{ticker}.keras")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    forecasting_model = refined_dnn_model(X_train_scaled)
         
    forecasting_model.fit(
        X_train_scaled, Y_train_scaled,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_scaled, Y_val_scaled),
        callbacks=[model_checkpoint_callback]
    )

    forecasting_model = tf.keras.models.load_model(checkpoint_filepath)
    loss, mae = forecasting_model.evaluate(X_test_scaled, Y_test_scaled, verbose=0)
    y_predict = forecasting_model.predict(X_test_scaled)

    y_predict_var = np.var(y_predict)
    Y_test_scaled_var = np.var(Y_test_scaled.values)

    result = pd.DataFrame({
        'y_predict': y_predict_var,
        'y_test': Y_test_scaled_var,
        "test_loss": loss,
        "test_mae": mae
    }, index=[1])

    test_results_dir = os.path.join(os.path.dirname(__file__), 'test_results')
    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir)
    test_results_file = os.path.join(test_results_dir, 'result.md')
    open_mode = 'a' if os.path.exists(test_results_file) else 'w'
    with open(test_results_file, open_mode) as f:
        if open_mode == 'w':
            print("| ticker | Predicted Variance | Actual Variance | Test Loss (MSE) | Test MAE |", file=f)
        print("|---|---|---|---|---|", file=f)
        print(f"| {ticker} | {y_predict_var:.6f} | {Y_test_scaled_var:.6f} | {loss:.6f} | {mae:.6f} |", file=f)
