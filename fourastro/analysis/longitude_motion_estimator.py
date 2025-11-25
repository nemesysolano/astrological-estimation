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
from .utils import clean_nan_and_inf

Dense, Dropout, BatchNormalization, Concatenate, Input = (tf.keras.layers.Dense, tf.keras.layers.Dropout, tf.keras.layers.BatchNormalization, tf.keras.layers.Concatenate, tf.keras.layers.Input)
Model = tf.keras.models.Model
AdamW = tf.keras.optimizers.AdamW
import numpy as np

LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
l2 = tf.keras.regularizers.l2

planet_name_pairs = combinations([planet[1] for planet in astro.planets], 2  )

astrological_varriables = ["A_0","A_1","A_2","A_3","A_4","A_5","A_6","A_7","A_8","A_9","A_10","A_11"]
financial_variables = ["Y_Low","Y_High","Y_Close","ATRP","BBW_Low","BBW_High","BBW_Close","RVO_Low","RVO_High","RVO_Close","relative_volume"]

def pct_difference(a, b):
    return 2*(b - a) / (a+b)

def define_astrological_x(historical_data): 
    result = astrological_varriables + financial_variables
    return historical_data[result]

def define_financial_x(historical_data):
    return historical_data[financial_variables]

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
    
    return Y

def define_variables(train_data, validation_data, test_data, column_name, define_X):        
    Y_train_unscaled = define_Y(train_data, column_name)
    Y_val_unscaled = define_Y(validation_data, column_name)
    Y_test_unscaled =define_Y(test_data, column_name)
    
    X_train_unscaled = define_X(train_data).loc[Y_train_unscaled.index]
    X_val_unscaled = define_X(validation_data).loc[Y_val_unscaled.index]
    X_test_unscaled = define_X(test_data).loc[Y_test_unscaled.index]

    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train_unscaled)
    X_val_scaled = X_scaler.transform(X_val_unscaled)
    X_test_scaled = X_scaler.transform(X_test_unscaled)
    
    Y_scaler =  MinMaxScaler( feature_range=(-1, 1))
    Y_train_scaled = Y_train_unscaled
    Y_val_scaled = Y_val_unscaled
    Y_test_scaled = Y_test_unscaled
    
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
    
def refined_dnn_astrological_model(X_train_scaled):
    set_all_seeds()
    input_dim = X_train_scaled.shape[1] 
    regularizer = l2(1e-4) # Define a small L2 penalty (  1e-4, 1e-5, AND 1e-6 )
    reg_fin = l2(1e-6)   # Low regularization for the "Safe" financial path
    reg_astro = l2(1e-4) # Higher regularization for the "Wild" astrological path       
# ... (initial setup)
    
    # ------------------ Input Layer ------------------
    input_tensor = Input(shape=(input_dim,))

    # Astrological features are the first 14 columns, market features are the rest
    astro_input = input_tensor[:, :len(astrological_varriables)]
    market_input = input_tensor[:, len(astrological_varriables):]

    # ------------------ Branch 1: Market Context (Strong Signal) ------------------
    # ADDED L2 REGULARIZATION
    market_branch = Dense(64, activation='relu', kernel_regularizer=reg_fin, name='market_feature_proc')(market_input)
    market_branch = BatchNormalization()(market_branch)

    # ------------------ Branch 2: Astrological Features (Weak/Complex Signal) ------------------
    # ADDED L2 REGULARIZATION
    astro_branch = Dense(64, activation='relu', kernel_regularizer=reg_astro, name='astro_feature_proc')(astro_input)
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
                  loss='mae', 
                  metrics=['mae'])
    
    return model

def refined_dnn_financial_model(X_train_scaled):    
    set_all_seeds()
    input_dim = X_train_scaled.shape[1] 
    regularizer = l2(1e-4) # Define a small L2 penalty (  1e-4, 1e-5, AND 1e-6 )
    
    # ------------------ Input Layer ------------------
    input_tensor = Input(shape=(input_dim,))
    
    # All inputs are market features
    market_input = input_tensor

    # ------------------ Branch 1: Market Context (Strong Signal) ------------------
    # ADDED L2 REGULARIZATION
    market_branch = Dense(16, activation='relu', kernel_regularizer=regularizer, name='market_feature_proc')(market_input)
    x = BatchNormalization()(market_branch) # x is now market_branch
    
    # Deep layers for cross-feature interaction
    # ADDED L2 REGULARIZATION
    x = Dense(64, activation='relu', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x) 

    # ADDED L2 REGULARIZATION
    x = Dense(32, activation='relu', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Output layer (tanh forces output to [-1, 1] range)
    output_tensor = Dense(1, activation='tanh')(x)
    
    # Define the full model
    model = Model(inputs=input_tensor, outputs=output_tensor)
    
    # Use AdamW optimizer for better regularization
    model.compile(optimizer=AdamW(learning_rate=0.001), 
                  loss='mae', 
                  metrics=['mae'])
    
    return model
    
def lag_relative_volume(data, window=1):
    data['relative_volume'] = data['relative_volume'].shift(window)
    data.dropna(inplace=True)
    return data
    
def longitude_motion_estimator(ticker, price, model):    
    define_model = refined_dnn_astrological_model if model == 'astro' else refined_dnn_financial_model
    define_X = define_astrological_x if model == 'astro' else define_financial_x
    
    data = lag_relative_volume(clean_nan_and_inf(load_market_data(ticker)))
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
    
    X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, _, _ = define_variables(train_data, validation_data, test_data, price, define_X)
    
    # Define ModelCheckpoint callback to save the best model
    checkpoint_filepath =  os.path.join(os.getcwd(), 'models', f"L-to-Y-{ticker}-{price}-{model}.keras")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        monitor='val_mae',
        mode='min'
    )

    forecasting_model = define_model(X_train_scaled)
         
    forecasting_model.fit(
        X_train_scaled, Y_train_scaled,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_scaled, Y_val_scaled),
        callbacks=[model_checkpoint_callback]
    )

    forecasting_model = tf.keras.models.load_model(checkpoint_filepath)
    _, mae = forecasting_model.evaluate(X_test_scaled, Y_test_scaled, verbose=0)
    y_predict = forecasting_model.predict(X_test_scaled)

    y_predict_var = np.var(y_predict)
    Y_test_scaled_var = np.var(Y_test_scaled.values)

    test_results_dir = os.path.join(os.getcwd(), 'test-results')
    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir)

    test_results_file = os.path.join(test_results_dir, f"{price}-{model}-result.md")
    open_mode = 'a' if os.path.exists(test_results_file) else 'w'

    with open(test_results_file, open_mode) as f:
        if open_mode == 'w':
            print("| ticker   | Predicted Variance  | Actual Variance         | Test MAE  |", file=f)
            print("|----------|---------------------|-------------------------|-----------|", file=f)
        print(   f"| {ticker} | {y_predict_var:.6f} | {Y_test_scaled_var:.6f} | {mae:.6f} |", file=f)
