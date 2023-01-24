import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout

train = pd.read_csv('../input/Updated_DataFrame_Train.csv')
test = pd.read_csv('../input/Updated_DataFrame_Test.csv')

train = train.drop(['Frequency_cltv'], axis=1)

cat_cols = [col for col in train.columns if train[col].dtypes == 'object']
for col in cat_cols:
    le = LabelEncoder()
    train.loc[:, col] = le.fit_transform(train[col])

cat_cols = [col for col in test.columns if test[col].dtypes == 'object']
for col in cat_cols:
    le = LabelEncoder()
    test.loc[:, col] = le.fit_transform(test[col])

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


def create_model(hidden_units, dropout_rate):
    model = Sequential()
    model.add(Dense(hidden_units, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


model = KerasRegressor(build_fn=create_model, verbose=100)

param_grid = {'hidden_units': [32, 64, 128], 'dropout_rate': [0.1, 0.2, 0.3]}
grid = GridSearchCV(estimator=model, scoring='r2', param_grid=param_grid, n_jobs=-1, cv=3, verbose=100)
grid_result = grid.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
