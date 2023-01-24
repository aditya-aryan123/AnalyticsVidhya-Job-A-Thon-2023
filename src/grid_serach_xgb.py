import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import model_selection

if __name__ == "__main__":
    train = pd.read_csv('../input/train_BRCpofr.csv')

    train['income'] = train['income'].replace({'<=2L': 'Less than or equal to 2L'})

    cat_cols = [col for col in train.columns if train[col].dtypes == 'object']
    train = pd.get_dummies(data=train, columns=cat_cols)

    X = train.drop("cltv", axis=1).values
    y = train.cltv.values

    regressor = xgb.XGBRegressor()
    param_grid = {
        "eta": [0.1, 0.3, 0.5, 0.7, 0.9],
        'max_depth': [3, 5, 7, 9, 11, 15, 20]
    }
    model = model_selection.GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        verbose=10,
        n_jobs=1,
        cv=3
    )
    model.fit(X, y)
    print(f"Best score: {model.best_score_}")
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")
