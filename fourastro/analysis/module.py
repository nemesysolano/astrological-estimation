
from market import load_market_data
import pandas as pd 
import astro
import mlx.core as mx
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def lambda_total(t):    
    return np.sum(np.array([astro.astro_constants[planet_name]['λ'][t] for planet_name in [planet[1] for planet in astro.planets]]))

def define_Y(price, p_ref, gravity_sum):
    b = ((price - p_ref) / gravity_sum).to_numpy()
    b.shape = (price.shape[0], 1)
    return b

def define_X(data, terms):
    cases = len(data)
    shape = (cases, terms*2 + 1)
    A = np.zeros(shape)
    case = 0
    for t in data.index:        
        λ_total = np.deg2rad(lambda_total(t))
        A[case, 0] = 1
        k = 1
        i = 1
        for j in range(0, terms):
            A[case, i] =  k * np.cos(λ_total)
            A[case, i+1] =  k * np.sin(λ_total)
            i += 2
            k += 1
        case += 1
    
    return A


def fit_to_range(n, x, y, a, b):
    return ((n - x) * (b - a) / (y - x)) + a

def forecast(symbol):    
    train_size = 0.7
    validate_size = 0.2
    data = load_market_data(symbol)
    data = data.dropna()
    data = data[np.isfinite(data).all(axis=1)]
    data_idx = data.index
  
    # Date range for train data
    train_data_start_date = data_idx[0]
    train_data_end_date = data_idx[int(len(data_idx) * train_size)]

    # Date range for validate data
    validate_data_start_date = train_data_end_date + pd.Timedelta(days=1)
    validate_data_end_date = validate_data_start_date + pd.Timedelta(days=int(len(data_idx) * validate_size))

    # Date range for test data
    test_data_start_date = validate_data_end_date + pd.Timedelta(days=int(len(data_idx) * validate_size) + 1)
    
    # Train, validate and test datasets
    train_data = data[:train_data_end_date]
    validate_data = data[validate_data_start_date:validate_data_end_date]
    test_data = data[test_data_start_date:]

    # Compute gravity sum 
    gravity_sum = np.sum([planet[2] / (planet[3] * planet[3]) for planet in astro.planets])
    
    # Compute reference price (aka mean price)
    p_ref = fit_to_range(train_data['Close'].mean(), train_data['High'].max(), train_data['Low'].min(), 0.0005, 1)
    
    # Define Scaler
    scaler_X = StandardScaler()
    scaler_Y = MinMaxScaler(feature_range=(0, 1))

    # X, Y for training, validating and testing
    T = 6
    X_train = define_X(train_data, T)
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_validate = define_X(validate_data, T)
    X_validate_scaled = scaler_X.transform(X_validate)
    X_test = define_X(test_data, T)
    X_test_scaled = scaler_X.transform(X_test)


    Y_train = define_Y(train_data['Close'], p_ref, gravity_sum)
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    Y_validate = define_Y(validate_data['Close'], p_ref, gravity_sum)    
    Y_validate_scaled = scaler_Y.transform(Y_validate)
    Y_test = define_Y(test_data['Close'], p_ref, gravity_sum)
    Y_test_scaled = scaler_Y.transform(Y_test)

    print(X_train.shape[0], X_validate.shape[0], X_test.shape[0])
    model = keras.Sequential([
        keras.layers.Dense(
            64,
            activation='relu',
            input_shape=(X_train.shape[1],),
            kernel_initializer='he_normal'
        ),
        keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
        keras.layers.Dense(1, kernel_initializer='glorot_uniform')
    ])

    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train_scaled, Y_train_scaled, epochs=10, batch_size=32, validation_data=(X_validate_scaled, Y_validate_scaled))

    loss = model.evaluate(X_test_scaled, Y_test_scaled)

    
       
