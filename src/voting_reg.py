import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold


def run():
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

    cat_cols = [col for col in train.columns if train[col].dtypes == 'object']
    for col in cat_cols:
        le = preprocessing.LabelEncoder()
        train.loc[:, col] = le.fit_transform(train[col])

    cat_cols = [col for col in test.columns if test[col].dtypes == 'object']
    for col in cat_cols:
        le = preprocessing.LabelEncoder()
        test.loc[:, col] = le.fit_transform(test[col])

    X = train.drop('cltv', axis=1)
    y = train['cltv']

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=5)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model1 = LinearRegression()
    model2 = Lasso(alpha=0.9, tol=0.01)
    model3 = GradientBoostingRegressor(learning_rate=0.1, max_depth=5)
    model4 = Ridge(alpha=0.9, tol=0.1)

    ensemble = VotingRegressor(estimators=[('lr', model1), ('gb', model2), ('la', model3), ('ri', model4)])
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    print("R2 Score: ", r2_score(y_test, y_pred))

    test['prediction'] = ensemble.predict(test)
    submission = test[['id', 'prediction']].copy()
    print(submission.shape)
    submission.rename(columns={'prediction': 'cltv'}, inplace=True)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    run()
