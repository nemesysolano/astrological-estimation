
import keras
import tensorflow as tf
import numpy as np

def estimate_polarization_params(Y_train_scaled):
    """
    Estimates optimal 'weight' and 'width' for the polarization loss function 
    based on the scaled target series (Y_train_scaled).

    The 'weight' is dynamically determined using a safety margin (SM) 
    that is inversely proportional to the variance of Y.
    """
    Y_train_scaled = np.asarray(Y_train_scaled).flatten()

    if len(Y_train_scaled) < 2:
        # Fallback if insufficient data
        return 0.9, 0.0025

    # --- 1. Optimal 'width' (omega) ---
    # Fixed to create a sharp, narrow 'forbidden zone' around 0 (the "deadband").
    optimal_width = 0.0025
    
    # --- 2. Dynamic 'weight' (W) ---
    # W must be strong enough to overcome the baseline MSE (which is approx. Var(Y)).
    baseline_mse_at_zero = np.var(Y_train_scaled)
    epsilon_var = 1e-4 
    
    # Dynamic Safety Margin: The factor required to boost the variance up to 1.0 (max squared error).
    # This aggressively penalizes mode collapse in low-variance stocks.
    estimated_safety_margin = 1.0 / (baseline_mse_at_zero + epsilon_var)

    # Calculate W: Cap the boosted variance at 1.0, and ensure a minimum floor.
    optimal_weight = min(1.0, estimated_safety_margin * baseline_mse_at_zero)
    optimal_weight = max(0.5, optimal_weight) 

    return optimal_weight, optimal_width

def get_polarization_loss(weight, width):
    """
    weight: Strength of the penalty for predicting near 0.
    width: Controls the variance of the Gaussian (how wide the 'forbidden zone' is).
    """
    @keras.saving.register_keras_serializable()
    def polarization_loss(y_true, y_pred):
        # 1. Standard Accuracy Term (MSE)
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # 2. Opinionated Penalty Term
        # This function creates a 'hill' at 0. 
        # If y_pred is 0, penalty is max. If y_pred is -1 or 1, penalty is near 0.
        penalty = tf.reduce_mean(weight * tf.exp(-tf.square(y_pred) / width))
        
        return mse + penalty
    
    return polarization_loss

# Usage:
# model.compile(optimizer='adam', loss=get_polarization_loss(weight=1.0, width=0.2))

@keras.saving.register_keras_serializable()
def magnitude_weighted_loss(y_true, y_pred):
    # Calculate standard squared error
    squared_error = tf.square(y_true - y_pred)
    
    # Create a weight based on the ACTUAL value.
    # We add a small epsilon (0.1) so we don't completely ignore flat markets.
    # When y_true is +/-1, weight is ~1.1. When y_true is 0, weight is 0.1.
    weights = tf.abs(y_true) + 0.5
    
    return tf.reduce_mean(weights * squared_error)

# Usage:
# model.compile(optimizer='adam', loss=magnitude_weighted_loss)

def get_sign_penalty_loss(penalty_weight=2.0):

    @keras.saving.register_keras_serializable()
    def sign_penalty_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Check for sign disagreement
        # y_true * y_pred will be negative if signs are different
        # We take the negative of that (making it positive) and apply ReLU
        sign_mismatch = tf.nn.relu(-(y_true * y_pred))
        
        return mse + (penalty_weight * tf.reduce_mean(sign_mismatch))
    
    return sign_penalty_loss

# Usage:
# model.compile(optimizer='adam', loss=get_sign_penalty_loss(penalty_weight=1.5))