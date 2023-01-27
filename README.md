# AnalyticsVidhya-Job-A-Thon-2023

Overview of the Competition
The objective of the hackathon was to develop a machine learning approach to predict the energy consumption hourly for the next 6 months. It was a time-series problem. The metrics used in the competition was root mean squared error. Dependent variable was 'Row Id', 'Date' and Target variable was 'Energy'.

My view on the Problem
Basic EDA revealed there were ~2% missing values in energy column, there was an upward trend in the yearly energy cosnumption. Extracted new date time features from the date column. Comparing the graph of hour and car demand according to that I noticed that in the morning(at 6 am) energy consumption increases, this could be due to the fact as most people wake up at 6 and. After that I applies cosine and sine transformation for cyclical features like hour, month and day_of_week. Then I calculated Variance Inflation factor using statsmodels library and found that year and is_quarter_date are the ones with most collinearity. So, I removed them. I experimented with the basic algorithms and found Linear Regression, Decision Tree Regression and Extra Tree Regressor to give the lowest root mean squared error. However, when tested against the ground truth on Analytics Vidhya, it performed poorly. After that I tried XGBoost Regressor, CatBoost Regressor and LightGBM. My experimentations revealed that the LightGBM model seemed to perform well. Therefore, I went ahead with the said model to test which of the hyperparameters gave the highest score as per the ground truth. Finally, I scripted the entire process in a modular fashion to create a pipeline that could be deployed and automated for future use.

File Structure
grid_search_xgb.py: Performed grid search on XGBoost parameters

lgm_grid_search.py: Performed grid search on LightGBM parameters

grid_search_cat_boost.py: Performed grid search on CatBoost parameters

model_xgb.py: XGBoost Regressor Model

lgbm_run.py: LightGBM Regressor Model

model_cat_boost.py: CatBoost Regressor Model

Final Result
There were 6388 people on the Hacka-Thon out of which I got the rank of: Public Leaderboard Rank: 141 Private Leaderboard Rank: 186

## Competition Link:

https://datahack.analyticsvidhya.com/contest/job-a-thon-january-2023/
