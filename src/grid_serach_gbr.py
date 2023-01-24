import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import model_selection

if __name__ == "__main__":
    train = pd.read_csv('../input/new_train_frame.csv')

    train = train[['id', 'gender', 'area', 'income', 'marital_status', 'claim_amount', 'num_policies',
                   'type_of_policy', 'education_level', 'avg_claim_per_id', 'no_of_policies_to_vintage_ratio',
                   'income_vintage', 'Frequency_claim_amount', 'Ratio_vintage_to_policy',
                   'cltv']]

    train['income'] = train['income'].replace({'<=2L': 'Less than or equal to 2L'})

    cat_cols = [col for col in train.columns if train[col].dtypes == 'object']
    train = pd.get_dummies(data=train, columns=cat_cols)

    X = train.drop("cltv", axis=1).values
    y = train.cltv.values

    regressor = GradientBoostingRegressor()
    param_grid = {
        "learning_rate": [0.1, 0.3, 0.5, 0.7, 0.9, 0.01, 0.03, 0.05, 0.07, 0.09],
        'max_depth': [3, 5, 7, 9, 11, 15, 20],
        'subsample': [0.5, 0.6, 0.7, 0.8],
        'min_samples_split': [2, 5, 10, 15]
    }
    model = model_selection.GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        scoring="r2",
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
