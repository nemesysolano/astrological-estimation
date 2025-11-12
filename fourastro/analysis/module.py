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

def define_X(Y_combined, atr): 
    astro_constants = astro.astro_constants
    X_astro = []
    
    # Start columns with the two lagged features
    columns = ['ψ_1', 'ψ_2', 'ATR'] 
    
    # Append the names for the astrological features
    columns.extend([ f"A_{i}" for i in range(len(astro.planets) * 2)])

    for t in Y_combined.index: 
        # NEW STEP: Prepend the two lagged Y values
        x = [Y_combined.loc[t, 'ψ_1'], Y_combined.loc[t, 'ψ_2'], atr[t]]
        
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
    
    # The target is still Y(t), which is column 'ψ'
    return X

def define_Y(dataset, column_name):
    # This assumes dataset is sorted by time ascendingly
    
    # Calculate Y(t)
    upper_one = lambda v: np.minimum(1, v)
    
    # We must align p_t with p_{t-1} and v_t with v_{t-1}
    # A single index array for the result Y(t)
    s_index = dataset.index[1:].copy() 
    
    # p_t: price at time t (starts from second element)
    current_price = dataset[column_name].copy().iloc[1:].values
    # p_{t-1}: price at time t-1 (starts from first element)
    previous_price = dataset[column_name].copy().iloc[:-1].values 
    
    # v_t: volume at time t
    current_volume = dataset['Volume'].copy().iloc[1:].values
    # v_{t-1}: volume at time t-1
    previous_volume = dataset['Volume'].copy().iloc[:-1].values

    price_pct_diff = pct_difference(previous_price, current_price)
    # The volume component of Y(t)
    volume_exp_diff = upper_one(current_volume / previous_volume) 
    
    # Y(t)
    Y = pd.DataFrame(price_pct_diff * volume_exp_diff, index=s_index, columns=['ψ'])
    
    # ----------------------------------------------------
    # NEW STEP: Create Y(t-1) and Y(t-2)
    Y_lag1 = Y['ψ'].shift(periods=1)
    Y_lag1.name = 'ψ_1'
    
    Y_lag2 = Y['ψ'].shift(periods=2) # NEW: Shift by 2 periods
    Y_lag2.name = 'ψ_2'
    
    # Combine Y(t), Y(t-1), and Y(t-2)
    Y_combined = pd.concat([Y, Y_lag1, Y_lag2], axis=1)
    
    # Remove the first two rows where Y(t-1) or Y(t-2) are NaN
    Y_combined.dropna(inplace=True)
    
    return Y_combined

def define_variables(train_data, validation_data, test_data, column_name):    
    Y_train_combined = define_Y(train_data, column_name)
    Y_val_combined = define_Y(validation_data, column_name)
    Y_test_combined = define_Y(test_data, column_name)

    Y_train_unscaled = Y_train_combined['ψ']
    Y_val_unscaled = Y_val_combined['ψ']
    Y_test_unscaled =Y_test_combined['ψ']

    X_train_unscaled = define_X(Y_train_combined, train_data['ATR'])
    X_val_unscaled = define_X(Y_val_combined, validation_data['ATR'])
    X_test_unscaled = define_X(Y_test_combined, test_data['ATR'])

    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train_unscaled)
    X_val_scaled = X_scaler.transform(X_val_unscaled)
    X_test_scaled = X_scaler.transform(X_test_unscaled)

    Y_scaler =  MinMaxScaler(feature_range=(0, 1))
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
    
def improved_dnn_model(X_train_scaled):
    set_all_seeds()
    # Number of input features is X_train_scaled.shape[1] (14 features)
    input_dim = X_train_scaled.shape[1] 
    
    model = tf.keras.Sequential([
        # Initial wide layer to capture complex interactions
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)), 
        tf.keras.layers.BatchNormalization(), # Add Batch Normalization
        tf.keras.layers.Dropout(0.1), # Increase dropout slightly
        
        # Deeper, progressively narrowing layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),

        # Final Feature Compression
        # Change Tanh to a final ReLU/SELU before output, or keep Tanh
        tf.keras.layers.Dense(64, activation='relu'), 
        tf.keras.layers.Dropout(0.1),
        
        # Output layer (Predicts Y(t) which is bounded [-1, 1] but usually close to 0)
        # Using Tanh on the output forces the prediction to be in [-1, 1], matching Y(t)'s definition
        tf.keras.layers.Dense(1, activation='tanh') 
    ])
    
    # Use a more sophisticated optimizer like Nadam or AdamW
    model.compile(optimizer='nadam', loss='mse', metrics=['mae']) #
    
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

    # Calculate Average True Range (ATR)
    data['ATR'] = true_range.ewm(span=window, adjust=False).mean()
    return data

def forecast(ticker):
    data = add_atr_to_dataframe(clean_nan_and_inf(load_market_data(ticker)))

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
    checkpoint_filepath = 'best_model.keras'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    forecasting_model = improved_dnn_model(X_train_scaled)
         
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

    y_predict_var = np.var(y_predict.flatten())
    Y_test_scaled_var = np.var(Y_test_scaled.to_numpy().flatten())

    result = pd.DataFrame({
        'y_predict': y_predict_var,
        'y_test': Y_test_scaled_var,
        "test_loss": loss,
        "test_mae": mae
    }, index=[1])

    print("| ticker | Predicted Variance | Actual Variance | Test Loss (MSE) | Test MAE |")
    print("|---|---|---|---|---|")
    print(f"| {ticker} | {y_predict_var:.6f} | {Y_test_scaled_var:.6f} | {loss:.6f} | {mae:.6f} |")
    print()
    