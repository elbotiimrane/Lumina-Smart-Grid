import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

class HistoricalAverageModel:
    """Historical Average (HA) model that predicts traffic based on historical mean for specific node, hour, and day-of-week."""
    def __init__(self):
        self.lookup = {}
        
    def fit(self, X_train, y_train):
        # We need the original Node_ID, Hour, and Day_of_Week which are in X_train
        # Let's combine them with y_train to compute the means
        df = X_train.copy()
        df['Target'] = y_train
        
        # Group by Node_ID, Hour, and Day_of_Week
        self.lookup = df.groupby(['Node_ID', 'Hour', 'Day_of_Week'])['Target'].mean().to_dict()
        self.global_mean = y_train.mean()
        
    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            key = (row['Node_ID'], row['Hour'], row['Day_of_Week'])
            predictions.append(self.lookup.get(key, self.global_mean))
        return np.array(predictions)

def get_temporal_features():
    return [
        'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow',
        'Is_Weekend', 'Is_Rush', 'Is_Night', 'Ramadan_Effect',
        'Ped_Lag_1', 'Veh_Lag_1', 'Ped_Lag_2', 'Veh_Lag_2',
        'Ped_Lag_3', 'Veh_Lag_3', 'Ped_Lag_48', 'Veh_Lag_48'
    ]

def get_spatio_temporal_features():
    return get_temporal_features() + [
        'degree_centrality', 'closeness_centrality', 'betweenness_centrality', 'eigenvector_centrality',
        'Spatial_Lag_Ped', 'Spatial_Lag_Veh'
    ]

def train_and_predict_all(df_train, df_test, target_col):
    """Trains and predicts all 5 benchmark models for a given target column (Pedestrians or Vehicles)."""
    y_train = df_train[target_col].values
    y_test = df_test[target_col].values
    
    predictions = {}
    
    # 1. Historical Average (HA)
    ha_model = HistoricalAverageModel()
    ha_model.fit(df_train, y_train)
    predictions['Historical Average'] = ha_model.predict(df_test)
    
    # 2. Linear Ridge Regression (Temporal-Only)
    temp_cols = get_temporal_features()
    ridge_temp = Ridge(alpha=1.0)
    ridge_temp.fit(df_train[temp_cols], y_train)
    predictions['Ridge (Temporal-Only)'] = ridge_temp.predict(df_test[temp_cols])
    
    # 3. Temporal XGBoost (Temporal-Only)
    xgb_temp = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_temp.fit(df_train[temp_cols], y_train)
    predictions['XGBoost (Temporal-Only)'] = xgb_temp.predict(df_test[temp_cols])
    
    # 4. Graph-Feature-Enhanced Ridge Regression (Spatio-Temporal)
    st_cols = get_spatio_temporal_features()
    ridge_st = Ridge(alpha=1.0)
    ridge_st.fit(df_train[st_cols], y_train)
    predictions['Ridge (Spatio-Temporal)'] = ridge_st.predict(df_test[st_cols])
    
    # 5. Graph-Feature-Enhanced XGBoost (Spatio-Temporal)
    xgb_st = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_st.fit(df_train[st_cols], y_train)
    predictions['XGBoost (Spatio-Temporal)'] = xgb_st.predict(df_test[st_cols])
    
    return predictions, y_test
