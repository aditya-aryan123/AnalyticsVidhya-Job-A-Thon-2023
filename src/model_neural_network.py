import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from sklearn.metrics import r2_score

train = pd.read_csv('../input/Updated_frame.csv')
test = pd.read_csv('../input/updated_test_frame.csv')

# train = train.drop(['Frequency_cltv'], axis=1)

# train['income'] = train['income'].replace({'<=2L': 'Less than or equal to 2L'})
# test['income'] = test['income'].replace({'<=2L': 'Less than or equal to 2L'})

cat_cols = [col for col in train.columns if train[col].dtypes == 'object']
train = pd.get_dummies(data=train, columns=cat_cols)

cat_cols = [col for col in test.columns if test[col].dtypes == 'object']
test = pd.get_dummies(data=test, columns=cat_cols)

num_cols = [col for col in train.columns if train[col].dtypes != 'object']
for col in num_cols:
    scaler = StandardScaler()
    train.loc[:, col] = scaler.fit_transform(train[col].values.reshape(-1, 1))

num_cols = [col for col in test.columns if test[col].dtypes != 'object']
for col in num_cols:
    scaler = StandardScaler()
    test.loc[:, col] = scaler.fit_transform(test[col].values.reshape(-1, 1))

X = train.drop('cltv', axis=1)
y = train['cltv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

adam = optimizers.Adam()
model.compile(loss='mean_squared_error', optimizer=adam)
model.fit(X_train, y_train, epochs=200, batch_size=126)

y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
print("R2 Score:", score)
