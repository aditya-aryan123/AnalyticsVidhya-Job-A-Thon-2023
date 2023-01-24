import numpy as np
import pandas as pd
from sklearn import model_selection


def create_folds(data):
    data['kfold'] = -1
    data = data.sample(frac=1).reset_index(drop=True)
    num_bins = int(np.floor(1 + np.log2(len(data))))
    data.loc[:, 'cltv_binned'] = pd.cut(data['cltv'], bins=num_bins, labels=False)
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.cltv_binned.values)):
        data.loc[v_, 'kfold'] = f
    data = data.drop('cltv_binned', axis=1)
    data.to_csv('../input/train_folds.csv', index=False)
    return data


if __name__ == "__main__":
    df = pd.read_csv('../input/Updated_frame.csv')
    df = create_folds(df)
