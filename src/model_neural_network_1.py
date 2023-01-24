import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import regularizers
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras import backend as K
import matplotlib.pyplot as plt


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


train = pd.read_csv('../input/new_train_frame.csv')
test = pd.read_csv('../input/new_test_frame.csv')

train = train[['id', 'gender', 'area', 'income', 'marital_status', 'claim_amount', 'num_policies', 'policy',
               'type_of_policy', 'education_level', 'avg_claim_per_id', 'no_of_policies_to_vintage_ratio',
               'premium_amt_to_income_ratio', 'income_vintage', 'Ratio_vintage_to_policy', 'Frequency_claim_amount',
               'cltv']]

test = test[['id', 'gender', 'area', 'income', 'marital_status', 'claim_amount', 'num_policies', 'policy',
             'type_of_policy', 'education_level', 'avg_claim_per_id', 'no_of_policies_to_vintage_ratio',
             'premium_amt_to_income_ratio', 'income_vintage', 'Ratio_vintage_to_policy', 'Frequency_claim_amount']]

train['income'] = train['income'].replace({'<=2L': 'Less than or equal to 2L'})
test['income'] = test['income'].replace({'<=2L': 'Less than or equal to 2L'})

cat_cols = [col for col in train.columns if train[col].dtypes == 'object']
train = pd.get_dummies(data=train, columns=cat_cols)

cat_cols = [col for col in test.columns if test[col].dtypes == 'object']
test = pd.get_dummies(data=test, columns=cat_cols)

num_cols = [col for col in train.columns if train[col].dtypes != 'object' and col not in ['id']]
for col in num_cols:
    scaler = StandardScaler()
    train.loc[:, col] = scaler.fit_transform(train[col].values.reshape(-1, 1))

num_cols = [col for col in test.columns if test[col].dtypes != 'object' and col not in ['id']]
for col in num_cols:
    scaler = StandardScaler()
    test.loc[:, col] = scaler.fit_transform(test[col].values.reshape(-1, 1))

X = train.drop('cltv', axis=1)
y = train['cltv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

model = Sequential()
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[coeff_determination])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
                    callbacks=[early_stopping])

y_pred_test = model.predict(X_test)
score_test = r2_score(y_pred_test, y_test)
print(f"R2 Score Test: {score_test}")

y_pred_val = model.predict(X_val)
score_val = r2_score(y_pred_val, y_val)
print(f"R2 Score Validation: {score_val}")

test['prediction'] = model.predict(test)
submission = test[['id', 'prediction']].copy()
submission.rename(columns={'prediction': 'cltv'}, inplace=True)
submission.to_csv('submission.csv', index=False)

plt.figure(figsize=(15, 7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()
