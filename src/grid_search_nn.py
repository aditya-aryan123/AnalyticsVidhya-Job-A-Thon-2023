import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
from sklearn.model_selection import GridSearchCV
from keras.regularizers import l1, l2
from keras.wrappers.scikit_learn import KerasRegressor


data = pd.read_csv("../input/train_BRCpofr.csv")

cat_cols = [col for col in data.columns if data[col].dtypes == 'object']
for col in cat_cols:
    le = LabelEncoder()
    data.loc[:, col] = le.fit_transform(data[col])

X = data.drop(columns=["cltv"], axis=1)
y = data["cltv"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=None)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)
X_test_poly = poly.transform(X_test)

param_grid = {
    'hidden_layer_sizes': [(64, 64, 64), (32, 32, 32), (128, 128, 128)],
    'activation': ['relu', 'tanh'],
    'learning_rate': [0.01, 0.001, 0.0001],
    'l1_reg': [0.0, 0.01, 0.1],
    'l2_reg': [0.0, 0.01, 0.1]
}


def create_model(hidden_layer_sizes, activation, learning_rate, l1_reg, l2_reg):
    model = Sequential()
    model.add(Dense(hidden_layer_sizes[0], activation=activation, input_shape=(X_train_poly.shape[1],),
                    kernel_regularizer=l1(l1_reg)))
    model.add(Dense(hidden_layer_sizes[1], activation=activation, kernel_regularizer=l2(l2_reg)))
    model.add(Dense(hidden_layer_sizes[2], activation=activation, kernel_regularizer=l2(l2_reg)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate),
                  metrics=[RootMeanSquaredError()])
    return model


# Create a Keras wrapper for scikit-learn
model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=126, verbose=0)

# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_result = grid.fit(X_train_poly, y_train)

# Print the best parameters and the corresponding R-squared score on the validation set
print("Best parameters: {}".format(grid_result.best_params_))
print("Best R-squared score: {:.4f}".format(grid_result.best_score_))

# Regularization
# Add dropout layer to the model
model.add(Dropout(0.2))
