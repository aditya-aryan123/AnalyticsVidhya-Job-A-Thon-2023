import pandas as pd
from sklearn import model_selection
from sklearn import metrics
from sklearn import linear_model
from sklearn import pipeline
from sklearn import preprocessing
import catboost as cat


def run():
    train = pd.read_csv('../input/train_BRCpofr.csv')
    test = pd.read_csv('../input/test_koRSKBP.csv')

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

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=1)

    reg = cat.CatBoostRegressor()
    pipe_lr = pipeline.Pipeline([('scaler', preprocessing.StandardScaler()), ('model', reg)])
    pipe_lr.fit(X_train, y_train)
    preds = pipe_lr.predict(X_test)

    r2 = metrics.r2_score(y_test, preds)
    print(f"Squared={r2}")

    test['prediction'] = pipe_lr.predict(test)
    submission = test[['id', 'prediction']].copy()
    submission.rename(columns={'prediction': 'cltv'}, inplace=True)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    run()
