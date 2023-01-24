import pandas as pd


'''def generate_arithmatic_features(data, columns, operations):
    """
    Generate arithmetic features by performing mathematical operations on specified columns.
    :param data: DataFrame containing the data.
    :param columns: list of columns to generate arithmetic features on
    :param operations: list of mathematical operations to perform (e.g. '+', '-', '*', '/')
    :return: DataFrame containing the transformed data with new features
    """
    arithmatic_features = pd.DataFrame()
    for column1 in columns:
        for column2 in columns:
            for operation in operations:
                if column1 != column2:
                    feature_name = column1 + operation + column2
                    arithmatic_features[feature_name] = data[column1].apply(eval(f"lambda x: x {operation}")) + data[
                        column2]
    return arithmatic_features'''


def generate_ratio_frequency_groupby_features(data, columns, target_col):
    """
    Generate ratio, frequency, and groupby features.
    :param data: DataFrame containing the data.
    :param columns: list of columns to generate ratio, frequency, and groupby features on
    :param target_col: the target column to groupby
    :return: DataFrame containing the transformed data with new features
    """
    ratio_features = pd.DataFrame()
    for column in columns:
        ratio_features[f'{column}_ratio'] = data[column] / data[column].sum()

    freq_features = pd.DataFrame()
    for column in columns:
        freq_features[f'{column}_freq'] = data.groupby(column)[target_col].transform('count') / len(data)

    groupby_features = data.groupby(target_col)[columns].agg(['mean', 'min', 'max'])
    groupby_features.columns = ['_'.join(col) for col in groupby_features.columns]

    return pd.concat([ratio_features, freq_features, groupby_features], axis=1)


def generate_ratio_frequency_groupby_features_cat(data, columns, target_col):
    """
    Generate ratio, frequency, and groupby features for categorical data
    :param data: DataFrame containing the data.
    :param columns: list of columns to generate ratio, frequency, and groupby features on
    :param target_col: the target column to groupby
    :return: DataFrame containing the transformed data with new features
    """
    ratio_features = pd.DataFrame()
    for column in columns:
        ratio_features[f'{column}_ratio'] = data[column] / data[column].sum()

    freq_features = pd.DataFrame()
    for column in columns:
        freq_features[f'{column}_freq'] = data.groupby(column)[target_col].transform('count') / len(data)

    groupby_features = pd.DataFrame()
    for column in columns:
        if data[column].dtype == 'object':
            groupby_features[f'{column}_count'] = data.groupby(column)[target_col].transform('count')
            groupby_features[f'{column}_mode'] = data.groupby(column)[target_col].transform(
                lambda x: x.value_counts().index[0])
            groupby_features[f'{column}_nunique'] = data.groupby(column)[target_col].transform('nunique')
        else:
            groupby_features = data.groupby(target_col)[columns].agg(['mean', 'min', 'max'])
            groupby_features.columns = ['_'.join(col) for col in groupby_features.columns]

    return pd.concat([ratio_features, freq_features, groupby_features], axis=1)


data = pd.read_csv('../input/train_BRCpofr.csv')
columns = data.columns.tolist()
'''operations = ['+', '-', '*', '/']
arithmatic_features = generate_arithmatic_features(data, columns, operations)
data = pd.concat([data, arithmatic_features], axis=1)'''
target_col = 'cltv'
ratio_freq_groupby_features = generate_ratio_frequency_groupby_features_cat(data, columns, target_col)
data = pd.concat([data, ratio_freq_groupby_features], axis=1)
ratio_freq_groupby_features_cat = generate_ratio_frequency_groupby_features_cat(data, columns, target_col)
data = pd.concat([data, ratio_freq_groupby_features_cat], axis=1)
