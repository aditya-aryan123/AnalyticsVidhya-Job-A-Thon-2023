import pandas as pd
import tensorflow as tf
from keras.layers import Input, Embedding, Flatten, Concatenate, BatchNormalization, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


data = pd.read_csv("../input/train_BRCpofr.csv")

X = data.drop(columns=["cltv"], axis=1)
y = data["cltv"]

numerical_features = ['id', 'marital_status', 'vintage', 'claim_amount']
categorical_features = ['gender', 'area', 'qualification', 'income', 'num_policies', 'policy', 'type_of_policy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=None)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(), categorical_features)])

X_train_transformed = preprocessor.fit_transform(X_train)
X_val_transformed = preprocessor.transform(X_val)
X_test_transformed = preprocessor.transform(X_test)

X_train_transformed = tf.convert_to_tensor(X_train_transformed, dtype=float)
X_val_transformed = tf.convert_to_tensor(X_val_transformed, dtype=float)
X_test_transformed = tf.convert_to_tensor(X_test_transformed, dtype=float)

input_layers = {}
for column in X_train_transformed.columns:
    input_layers[column] = Input(shape=(1,), name=column)

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("embedding", Embedding(len(X_train_transformed[column].unique()), 8)),
    ("flatten", Flatten()),
    ("batch_normalization", BatchNormalization()),
    ("concatenate", Concatenate()),
    ("dense1", Dense(32, activation='relu')),
    ("dense2", Dense(64, activation='relu')),
    ("dense3", Dense(64, activation='relu')),
    ("output", Dense(1))
])


''''y_train = np.array(y_train, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)
X_train = np.array(X_train, dtype=np.float32)
X_test = np.asarray(X_test, dtype=np.float32)'''
model_pipeline.fit(X_train_transformed, y_train, epochs=100, batch_size=32, validation_split=0.2)

y_pred = model_pipeline.predict(X_test)
score = r2_score(y_test, y_pred)
print("R Score: ", score)
