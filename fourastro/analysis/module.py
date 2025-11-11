
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

def define_X(Y): # (A=constant, t=current time, T=Period, N=Numbert of terms, )    
    astro_constants = astro.astro_constants
    X = []
    columns = [ f"A_{i}" for i in range(len(astro.planets) * 2)]

    for t in Y.index:        
        x = []
        k = 1
        for planet in astro.planets:
            planet_name = planet[1]
            λ = astro_constants[planet_name]['λ'][t]
            a = astro_constants[planet_name]['g']
            b = astro_constants[planet_name]['b']
            T = astro_constants[planet_name]['T']
            f = 2 * k * np.pi /T            
            x.append(a * np.cos(f * λ))
            x.append(b * np.sin(f * λ))
            k+=1

        X.append(x)        

    X = pd.DataFrame(X, index=Y.index, columns=columns)
    return X

def define_Y(dataset, column_name):
    upper_one = lambda v: np.minimum(1, v)
                                                      
    s_index = dataset.index[:-1].copy()
    current_price = dataset[column_name].copy().iloc[:-1].values
    previous_price = dataset[column_name].copy().iloc[1:].values
    current_volume = dataset['Volume'].copy().iloc[:-1].values
    previous_volume = dataset['Volume'].copy().iloc[1:].values

    price_pct_diff = pct_difference(previous_price, current_price)
    volume_exp_diff = upper_one(previous_volume / current_volume)
    Y = pd.DataFrame(price_pct_diff *volume_exp_diff , index = s_index, columns=['ψ'])
    return Y 

def define_variables(train_data, validation_data, test_data, column_name):    
    Y_train_unscaled = define_Y(train_data, column_name)
    Y_val_unscaled = define_Y(validation_data, column_name)
    Y_test_unscaled = define_Y(test_data, column_name)

    X_train_unscaled = define_X(Y_train_unscaled)
    X_val_unscaled = define_X(Y_val_unscaled)
    X_test_unscaled = define_X(Y_test_unscaled)

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

def create_dnn_model(X_train_scaled):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],),kernel_regularizer=l2(0.0005)), # <-- Add L2 here
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(32, activation='relu',kernel_regularizer=l2(0.0005)), # <-- Add L2 here
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(16, activation='relu',kernel_regularizer=l2(0.0005)), # <-- Add L2 here
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model 

    
def forecast(ticker):
    data = clean_nan_and_inf(load_market_data(ticker))

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

    forecasting_model = create_dnn_model(X_train_scaled)
         
    forecasting_model.fit(
        X_train_scaled, Y_train_scaled,
        epochs=100,
        batch_size=16,
        validation_data=(X_val_scaled, Y_val_scaled),
        callbacks=[model_checkpoint_callback]
    )

    forecasting_model = tf.keras.models.load_model(checkpoint_filepath)
    loss, mae = forecasting_model.evaluate(X_test_scaled, Y_test_scaled, verbose=0)
    y_predict = forecasting_model.predict(X_test_scaled)

    print("y_predict_var", y_predict.var())
    print("y_test_var", Y_test_scaled.var())
    print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

    