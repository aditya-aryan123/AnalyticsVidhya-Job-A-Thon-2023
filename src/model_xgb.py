import pandas as pd
from sklearn import model_selection
from sklearn import metrics
from sklearn import pipeline
from sklearn import preprocessing
import xgboost as xgb


def run():
    train = pd.read_csv('../input/Updated_DataFrame_Train.csv')
    test = pd.read_csv('../input/Updated_DataFrame_Test.csv')

    train['income'] = train['income'].replace({'<=2L': 'Less than or equal to 2L'})
    test['income'] = test['income'].replace({'<=2L': 'Less than or equal to 2L'})

    cat_cols = [col for col in train.columns if train[col].dtypes == 'object']
    train = pd.get_dummies(data=train, columns=cat_cols)
    print(train.columns)

    cat_cols = [col for col in test.columns if test[col].dtypes == 'object']
    test = pd.get_dummies(data=test, columns=cat_cols)
    print(test.columns)

    X = train.drop('cltv', axis=1)
    y = train['cltv']

    skf = model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=5)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    reg = xgb.XGBRegressor(eta=0.1, max_depth=5)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)

    r2 = metrics.r2_score(y_test, preds)
    print(f"Squared={r2}")

    test['prediction'] = reg.predict(test)
    submission = test[['id', 'prediction']].copy()
    submission.rename(columns={'prediction': 'cltv'}, inplace=True)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    run()
