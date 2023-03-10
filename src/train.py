import pandas as pd
from sklearn import metrics
import argparse
import model_dispatcher
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def run(fold, model, model_type):
    if model_type == 'tree':
        df = pd.read_csv('../input/train_folds.csv')

        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        x_train = df_train.drop('cltv', axis=1).values
        y_train = df_train.cltv.values
        x_valid = df_valid.drop('cltv', axis=1).values
        y_valid = df_valid.cltv.values

        reg = model_dispatcher.models[model]
        reg.fit(x_train, y_train)
        preds = reg.predict(x_valid)

        rmse = metrics.mean_squared_error(y_valid, preds, squared=False)
        r2 = metrics.r2_score(y_valid, preds)
        print(f"Fold={fold}, Root Mean Squared Error={rmse}, R Squared={r2}")

    else:
        df = pd.read_csv('../input/train_folds.csv')

        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        x_train = df_train.drop('cltv', axis=1).values
        y_train = df_train.cltv.values
        x_valid = df_valid.drop('cltv', axis=1).values
        y_valid = df_valid.cltv.values

        reg = model_dispatcher.models[model]
        pipeline = Pipeline([('scaler', RobustScaler()), ('model', reg)])
        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_valid)

        rmse = metrics.mean_squared_error(y_valid, preds, squared=False)
        r2 = metrics.r2_score(y_valid, preds)
        print(f"Fold={fold}, Root Mean Squared Error={rmse}, Squared={r2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    parser.add_argument(
        "--model_type",
        type=str
    )
    args = parser.parse_args()
    run(
        fold=args.fold,
        model=args.model,
        model_type=args.model_type
    )
