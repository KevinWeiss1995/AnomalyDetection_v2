from tensorflow import keras
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

'''
A simple model with two hidden layers and a sigmoid output layer. 
The model uses dropout and L2 regularization to prevent overfitting.
Binary classification problem.
'''

def create_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(input_shape,), kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model