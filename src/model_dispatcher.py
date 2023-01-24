from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
import xgboost as xgb
import catboost as cat
import lightgbm as lgm

models = {
    'ridge': linear_model.Ridge(),
    'lasso': linear_model.Lasso(),
    'decision_tree': tree.DecisionTreeRegressor(),
    'random_forest': ensemble.RandomForestRegressor(),
    'linear_reg': linear_model.LinearRegression(),
    'etr': ensemble.ExtraTreesRegressor(),
    'gbr': ensemble.GradientBoostingRegressor(),
    'hgbr': ensemble.HistGradientBoostingRegressor(),
    'abr': ensemble.AdaBoostRegressor(),
    'xgb_regressor': xgb.XGBRegressor(),
    'cat': cat.CatBoostRegressor(),
    'lgm': lgm.LGBMRegressor()
}
