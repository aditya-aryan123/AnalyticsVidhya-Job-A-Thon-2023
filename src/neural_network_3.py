import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Input, Embedding, Flatten, Concatenate, BatchNormalization, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


data = pd.read_csv("../input/train_BRCpofr.csv")
X = data.drop(columns=["cltv"], axis=1)
y = data["cltv"]

preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), numerical_features), ("cat", OneHotEncoder(), categorical_features)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

X_train = tf.convert_to_tensor(X_train, dtype=float)
X_test = tf.convert_to_tensor(X_test, dtype=float)
y_train = tf.convert_to_tensor(y_train, dtype=float)
y_test = tf.convert_to_tensor(y_test, dtype=float)

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
from keras.models import Model

# create input layer
input_layer = Input(shape=(23,1))

# create convolutional layers
conv1 = Conv1D(64, kernel_size=3, activation='relu')(input_layer)
pool1 = MaxPooling1D(pool_size=2)(conv1)
conv2 = Conv1D(128, kernel_size=3, activation='relu')(pool1)
pool2 = MaxPooling1D(pool_size=2)(conv2)

# flatten the output of convolutional layers
flat = Flatten()(pool2)

# create dense layers
dense1 = Dense(64, activation='relu')(flat)
dense2 = Dense(32, activation='relu')(dense1)
output = Dense(1)(dense2)

# create the model
model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

# reshape the input data to 3D array for CNN
X_train_cnn = X_train_transformed.reshape(X_train_transformed.shape[0], X_train_transformed.shape[1], 1)
X_val_cnn = X_val_transformed.reshape(X_val_transformed.shape[0], X_val_transformed.shape[1], 1)
X_test_cnn = X_test_transformed.reshape(X_test_transformed.shape[0], X_test_transformed.shape[1], 1)

# train the model
model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_data=(X_val_cnn, y_val))