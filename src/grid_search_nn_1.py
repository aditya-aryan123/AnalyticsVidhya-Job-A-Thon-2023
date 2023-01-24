import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout

train = pd.read_csv('../input/train_BRCpofr.csv')
test = pd.read_csv('../input/test_koRSKBP.csv')

train = train.drop(['Frequency_cltv'], axis=1)

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


def create_model(optimizer='adam', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(60, input_dim=X_train.shape[1], kernel_initializer=init, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30, kernel_initializer=init, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(15, kernel_initializer=init, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer=init, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])
    return model


model = KerasRegressor(build_fn=create_model, verbose=100)

batch_size = [32, 64, 128]
epochs = [50, 100, 150]

param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
