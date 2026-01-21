from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import numpy as np
import tensorflow as tf
import random

def set_seed(seed=42):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_lstm_model(input_shape, loss="Huber", seed=42):
    """
    Builds a multivariate LSTM model for the MSAR framework.
    input_shape: (30, 4) -> 30 days lookback, 4 features
    seed: random seed for reproducibility
    """
    set_seed(seed)
    
    model = Sequential()
    # input_shape[0] is timesteps, input_shape[1] is number of features
    model.add(LSTM(units=50, input_shape=input_shape))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=0.001)
    
    if loss == "Huber":
        model.compile(optimizer=optimizer, loss=Huber(delta=0.01))
    else:
        model.compile(optimizer=optimizer, loss="mse")
    
    return model

def prepare_lstm_data(data, lookback=8):
    """
    Prepare time series data for LSTM training
    data: (n_days, n_stocks) array
    lookback: number of historical days to use for prediction
    Returns: X (samples, lookback, n_stocks), y (samples, n_stocks)
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])  # Past 'lookback' days
        y.append(data[i])  # Next day's return
    return np.array(X), np.array(y)