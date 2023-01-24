import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn import linear_model
from sklearn import pipeline
from sklearn import preprocessing


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
    train = pd.get_dummies(data=train, columns=cat_cols)

    cat_cols = [col for col in test.columns if test[col].dtypes == 'object']
    test = pd.get_dummies(data=test, columns=cat_cols)

    X = train.drop('cltv', axis=1)
    y = train['cltv']

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=1)

    reg = linear_model.LinearRegression()
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
