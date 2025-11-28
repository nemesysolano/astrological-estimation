import os
# Example import change (adjust path as needed)
from fourastro.analysis.metrics import estimate_polarization_params, get_polarization_loss, magnitude_weighted_loss
from market import load_market_data
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from itertools import combinations
from .utils import clean_nan_and_inf

Dense, Dropout, BatchNormalization, Concatenate, Input, Reshape = ( tf.keras.layers.Dense, tf.keras.layers.Dropout, tf.keras.layers.BatchNormalization, tf.keras.layers.Concatenate, tf.keras.layers.Input, tf.keras.layers.Reshape)

Sequential =  tf.keras.Sequential
Model = tf.keras.models.Model
AdamW = tf.keras.optimizers.AdamW
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
l2 = tf.keras.regularizers.l2
TimeDistributed = tf.keras.layers.TimeDistributed
Conv1D = tf.keras.layers.Conv1D
AdamW = tf.keras.optimizers.AdamW

angular_indicators = sum([[f"cos_θ{i}", f"sin_θ{i}"] for i in range(1, 5)], [])
structural_indicators = ["Y_Low", "Y_High", "Y_Close", "structural_direction"] # , "slow_trend_run", "fast_trend_run"
oscilator_indicators = ["ATRP", "BBW_Low","BBW_High","BBW_Close","RVO_Low","RVO_High","RVO_Close","RV", "RVI_Low","RVI_High","RVI_Close"]
# --- Define Model Parameters (Use same values as before) ---
TIMESTEPS = 14  # n, the lookback period (e.g., 14 bars)
tf.random.set_seed(42) # Initiate with the same seed

def pct_difference(a, b):
    return 2*(b - a) / (a+b)

def define_Y(historical_data, price):    
    return historical_data[price].shift(-1)
    
def define_gann_X(historical_data, index):
    features = [*angular_indicators, *oscilator_indicators] 
    return historical_data.loc[index, features]

    # fourier_terms = historical_data[angular_indicators].sum(axis = 1)
    # features = [*oscilator_indicators, *structural_indicators] 
    # X = pd.DataFrame(historical_data.loc[index, features])
        
def define_linear_X(historical_data, index):
    """
    Creates a DataFrame with time-lagged features for Y_Close, Y_High, and Y_Low.
    Each column Y_Close_k will contain the close price from k days ago.
    """
    X = pd.DataFrame(index=historical_data.index)
    rvo_high = historical_data.loc[index, 'RVO_High']
    rvo_low = historical_data.loc[index, 'RVO_Low']
    X["structural_direction"] = historical_data["structural_direction"]

    for k in range(0, int(len(angular_indicators) / 2), 2): 
        X[angular_indicators[k]] = historical_data.loc[index, angular_indicators[k]] * rvo_high
        X[angular_indicators[k+1]] = historical_data.loc[index, angular_indicators[k]] * rvo_low        
    return X





def create_convlstm_Y_gann_model(Y_train, output_name, n_features):
    # Define the number of features for each input branch
    n_angular_features = len(angular_indicators)
    n_structural_features = len(structural_indicators)

    # --- Define Input Layers ---
    # Main input that will be split
    main_input = Input(shape=(TIMESTEPS, n_features), name='main_input')

    # --- Angular Features Submodel ---
    angular_input = main_input[:, :, :n_angular_features]
    x1 = Dense(64, kernel_regularizer=l2(0.001), name='angular_dense')(angular_input)
    x1 = BatchNormalization(name='angular_bn')(x1)
    angular_branch = tf.keras.layers.Activation('relu', name='angular_relu')(x1)

    # --- Structural Features Submodel ---
    structural_input = main_input[:, :, n_angular_features:]
    x2 = Dense(64, kernel_regularizer=l2(0.001), name='structural_dense')(structural_input)
    x2 = BatchNormalization(name='structural_bn')(x2)
    structural_branch = tf.keras.layers.Activation('relu', name='structural_relu')(x2)

    # --- Merge Submodels ---
    merged_features = Concatenate(name='merged_features')([angular_branch, structural_branch])

    # --- Main LSTM Model ---
    # 1. LSTM Layer
    lstm_out = LSTM(units=64, return_sequences=False)(merged_features)
    # 2. Dropout Layer
    dropout1 = Dropout(0.3)(lstm_out)
    # 3. Dense Layer
    dense1 = Dense(units=32, activation='relu')(dropout1)
    dropout2 = Dropout(0.3)(dense1)
    # 4. Output Layer
    output = Dense(units=1, activation='tanh', name=output_name)(dropout2)

    model = Model(inputs=main_input, outputs=output, name=f"Functional_LSTM_for_{output_name}")

    weight, width = estimate_polarization_params(Y_train) 
    loss_metric = get_polarization_loss(weight, width)
    model.compile(
        optimizer='adam',
        loss=loss_metric,
        metrics=['mae', loss_metric]
    )

    return model

def create_convlstm_Y_tanh_model(Y_train_data, output_name, n_features):
    model = Sequential(name=f"LSTM_Predictor_for_{output_name}")
    timesteps = 14
    # Set seed for reproducibility as per previous pattern
    tf.random.set_seed(42)
    
    # 1. LSTM Layer (Sequential Data Processing)
    # The input shape is (TIMESTEPS, n_features)
    model.add(LSTM(units=128, return_sequences=False, kernel_regularizer=None, input_shape=(timesteps, n_features)))
    model.add(BatchNormalization())
    model.add(Dropout(0.075))

    # 2. Dense Layer (Feature Combination)
    model.add(Dense(units=128, activation='relu', kernel_regularizer=None)) 
    model.add(BatchNormalization())
    model.add(Dropout(0.075))

    # 3. Output Layer (Tanh activation forces output between -1 and 1)
    model.add(Dense(units=1, activation='tanh', name=output_name))

    # --- Dynamic Loss Compilation ---
    weight, width = estimate_polarization_params(Y_train_data) 
    
    # Log the determined parameters
    print(f"Model: {output_name} | Polarization Loss Parameters: Weight={weight:.4f}, Width={width:.4f}")
    
    model.compile(
        optimizer=AdamW(learning_rate=1e-3), 
        loss=magnitude_weighted_loss,
        metrics=['mae']
    )

    return model

def split_into_train_val_test(X, Y):
    """
    Splits the data into training, validation, and test sets.
    """
    # Combine X and Y to ensure rows are dropped consistently across both
    data = pd.concat([X, Y], axis=1)

    # Clean NaN and Inf values which might be present
    clean_nan_and_inf(data)

    # Separate X and Y again
    Y_clean = data[Y.name]
    X_clean = data.drop(columns=[Y.name])

    # Define split points (70% train, 20% validation, 10% test)
    n_samples = len(X_clean)
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.9)

    # Split the data chronologically
    X_train, Y_train = X_clean.iloc[:train_end], Y_clean.iloc[:train_end]
    X_val, Y_val = X_clean.iloc[train_end:val_end], Y_clean.iloc[train_end:val_end]
    X_test, Y_test = X_clean.iloc[val_end:], Y_clean.iloc[val_end:]

    # --- Normalize the data ---
    # 1. Normalize X (Features)
    X_scaler = StandardScaler()    
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_val_scaled = X_scaler.transform(X_val)    
    X_test_scaled = X_scaler.transform(X_test)
    
    # 2. Normalize Y (Target) to [-1, 1] for tanh activation
    Y_scaler = MinMaxScaler(feature_range=(-1, 1))
    Y_train_scaled = Y_train.values.reshape(-1, 1)
    Y_val_scaled = Y_val.values.reshape(-1, 1)
    Y_test_scaled = Y_test.values.reshape(-1, 1)

    # Return scaled data along with scalers to inverse transform predictions later
    return (X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, 
            X_test_scaled, Y_test_scaled, X_scaler, Y_scaler)

def create_datasets(ticker, define_X, predictor):
    historical_data = load_market_data(ticker)
    Y = define_Y(historical_data, f"Y_{predictor}")
    X = define_X(historical_data, Y.index)
    return split_into_train_val_test(X, Y)

# ... (rest of the file content before analyze function)

def create_sequences(X_data, Y_data, time_steps=1):
    Xs, Ys = [], []
    for i in range(len(X_data) - time_steps):
        v = X_data[i:(i + time_steps)]
        Xs.append(v)
        # Ys is appended with the target *after* the sequence: Y_data[i + time_steps]
        Ys.append(Y_data[i + time_steps])
    return np.array(Xs), np.array(Ys)

def discretize_predictions(y_pred, threshold=0.3):
    """
    Snaps predictions to -1, 0, or 1 based on a confidence threshold.
    """
    results = []
    for p in y_pred:
        if p > threshold:
            results.append(1)
        elif p < -threshold:
            results.append(-1)
        else:
            results.append(0)
    return np.array(results)

# In your analyze function:
# Y_pred = discretize_predictions(best_model.predict(X_test_reshaped), threshold=0.3)

def analyze(ticker, predictor, mode):
    define_X = define_gann_X if mode == 'gann' else define_linear_X
    (X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, 
     X_test_scaled, Y_test_scaled, X_scaler, Y_scaler) = create_datasets(ticker, define_X, predictor)
    
    n_features = X_train_scaled.shape[1]
    model_factory = create_convlstm_Y_gann_model if mode == 'gann' else create_convlstm_Y_tanh_model
    model =  model_factory(Y_train_scaled, f"Y_{predictor}", n_features)

    # Reshape X and Y data for Conv1D-LSTM input (samples, timesteps, features)
    X_train_reshaped, Y_train_reshaped = create_sequences(X_train_scaled, Y_train_scaled, TIMESTEPS)
    X_val_reshaped, Y_val_reshaped = create_sequences(X_val_scaled, Y_val_scaled, TIMESTEPS)
    X_test_reshaped, Y_test_reshaped = create_sequences(X_test_scaled, Y_test_scaled, TIMESTEPS)

    # Define ModelCheckpoint callback to save the best model
    checkpoint_filepath = os.path.join(os.getcwd(), 'models', f'{ticker}_{predictor}_{mode}.keras')

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        monitor='mae', # mae'val_magnitude_weighted_loss',        
        mode='min')

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor= 'mae', # val_magnitude_weighted_loss',
        patience=50,
        mode='min',
        restore_best_weights=True
    )

    # FIX: Use Y_train_reshaped and Y_val_reshaped as targets for model.fit
    model.fit(
        X_train_reshaped, 
        Y_train_reshaped,  # Corrected target array
        epochs=100, 
        batch_size=32, 
        validation_data=(X_val_reshaped, Y_val_reshaped), # Corrected validation target array
        callbacks=[model_checkpoint_callback, early_stopping_callback]
    )
    
    # Load the best model
    best_model = tf.keras.models.load_model(checkpoint_filepath)

    # Evaluate the model on the test set
    mae, mse, = best_model.evaluate(X_test_reshaped, Y_test_reshaped, verbose=0)

    # Make predictions on the test set
    Y_pred = best_model.predict(X_test_reshaped)
    Y_actual = Y_test_reshaped

    # Lower this to 0.1 to see if the variances come back to life
    Y_actual_var = np.var(Y_actual)
    Y_pred_var = np.var(Y_pred)

    variance_percentage_diff = Y_pred_var / Y_actual_var

    # --- Results Summary ---
    
    output_file = os.path.join(os.getcwd(), "test-results", f"{predictor}_{mode}.md")
    mode = 'a' if os.path.exists(output_file) else 'w'
    headers = ["Ticker", "Loss", "Test Mean Absolute Error", "Variance of Actual", "Variance of Predicted", "[Var Pred.] / [Var Act.]"]

    with open(output_file, mode) as f:
        values = [ticker, f"{mse:.8f}", f"{mae:.8f}", f"{Y_actual_var:.8f}", f"{Y_pred_var:.8f}", f"{variance_percentage_diff:.2f}"]

        if mode == 'w':
            # Determine column widths for formatting
            col_widths = [max(len(str(h)), len(str(v))) for h, v in zip(headers, values)]
            f.write("| " + " | ".join([h.ljust(w) for h, w in zip(headers, col_widths)]) + " |\n")
            f.write("|-" + "-|-".join(["-" * w for w in col_widths]) + "-|\n")
        col_widths = [max(len(h), len(v)) for h, v in zip(headers, values)]
        f.write("| " + " | ".join([v.ljust(w) for v, w in zip(values, col_widths)]) + " |\n")