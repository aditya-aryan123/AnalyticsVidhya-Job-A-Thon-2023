import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.metrics import RootMeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import optimizers

train = pd.read_csv('../input/new_train_frame.csv')
test = pd.read_csv('../input/new_test_frame.csv')

train = train[['id', 'gender', 'area', 'income', 'marital_status', 'claim_amount', 'num_policies',
               'type_of_policy', 'education_level', 'avg_claim_per_id', 'no_of_policies_to_vintage_ratio',
               'income_vintage', 'Frequency_claim_amount', 'Ratio_vintage_to_policy',
               'cltv']]

test = test[['id', 'gender', 'area', 'income', 'marital_status', 'claim_amount', 'num_policies',
             'type_of_policy', 'education_level', 'avg_claim_per_id', 'no_of_policies_to_vintage_ratio',
             'income_vintage', 'Frequency_claim_amount', 'Ratio_vintage_to_policy']]

train['income'] = train['income'].replace({'<=2L': 'Less than or equal to 2L'})
test['income'] = test['income'].replace({'<=2L': 'Less than or equal to 2L'})

X = train.drop(columns=["cltv"], axis=1)
y = train["cltv"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=None)

numerical_features = [col for col in X.columns if X[col].dtypes != 'object']
categorical_features = [col for col in X.columns if X[col].dtypes == 'object']
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

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_transformed.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
adam = optimizers.Adam()
model.compile(loss='mean_squared_error', optimizer=adam, metrics=[RootMeanSquaredError()])

model.fit(X_train_transformed, y_train, epochs=200, batch_size=24, validation_data=(X_val_transformed, y_val),
          callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

y_pred_val = model.predict(X_val_transformed)
score_val = r2_score(y_pred_val, y_val)
y_pred_test = model.predict(X_test_transformed)
score_test = r2_score(y_pred_test, y_test)
print("R2 Score Test:", score_test)

'''test['prediction'] = model.predict(test)
submission = test[['id', 'prediction']].copy()
submission.rename(columns={'prediction': 'cltv'}, inplace=True)
submission.to_csv('submission.csv', index=False)'''
