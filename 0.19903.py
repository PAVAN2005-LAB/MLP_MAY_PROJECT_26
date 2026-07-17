# -*- coding: utf-8 -*-
"""
Kaggle Notebook Solution: Heavy Equipment Price Prediction
Model: Multi-Seed LightGBM (Pure Feature Engineering Focus)
Target: RMSLE < 0.19
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

print("Initializing CPU-Optimized Multi-Seed Pipeline (Feature Focused)...")
train_path = '/kaggle/input/competitions/heavy-equipment-selling-price-prediction-challenge/train.csv'
test_path = '/kaggle/input/competitions/heavy-equipment-selling-price-prediction-challenge/test.csv'

train_df = pd.read_csv(train_path, low_memory=False)
test_df = pd.read_csv(test_path, low_memory=False)
target_col = 'TargetValue'

# ==========================================
# 1. ADVANCED FEATURE ENGINEERING
# ==========================================
def engineer_features(df_in):
    df = df_in.copy()
    
    # Parse Dates
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
    df['Trans_Year'] = df['TransactionDate'].dt.year
    df['Trans_Month'] = df['TransactionDate'].dt.month
    df['Trans_DayOfWeek'] = df['TransactionDate'].dt.dayofweek
    
    # Cyclic Time Transformation (Captures seasonal auction patterns)
    df['Month_sin'] = np.sin(2 * np.pi * df['Trans_Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Trans_Month'] / 12)
    
    # Handle Year Anomaly
    df['ManufactureYear_is_placeholder'] = (df['ManufactureYear'] == 1001).astype(int)
    df.loc[df['ManufactureYear'] == 1001, 'ManufactureYear'] = np.nan
    df['MachineAge'] = (df['Trans_Year'] - df['ManufactureYear']).clip(lower=0)
    
    # Operational Hours Winsorization
    df['OperationalHoursMeter'] = df['OperationalHoursMeter'].clip(upper=60000)
    df['LogHours'] = np.log1p(df['OperationalHoursMeter'])
    
    # Usage Intensity
    df['Usage_Intensity'] = df['OperationalHoursMeter'] / (df['MachineAge'] + 1)
    df['LogUsage_Intensity'] = np.log1p(df['Usage_Intensity'])
    
    # Multiplicative Multi-Tier Interaction
    df['Age_x_LogHours'] = df['MachineAge'] * df['LogHours']
    
    # Categorical Structural Compound Key
    df['FuncClass_x_InvGroup'] = df['FunctionalClassification'].astype(str) + '_' + df['InventoryGroupDescription'].astype(str)
    
    return df

print("Transforming core data structures...")
train_eng = engineer_features(train_df)
test_eng = engineer_features(test_df)

y = np.log1p(train_eng[target_col])
drop_cols = ['TransactionID', 'AssetID', 'TransactionDate', 'TargetValue', 'col18', 'col19']

# ==========================================
# 2. OUT-OF-FOLD (OOF) SMOOTH TARGET ENCODING
# ==========================================
target_enc_cols = ['FunctionalClassification', 'RegionCode', 'FuncClass_x_InvGroup']
SMOOTH = 25
global_mean = y.mean()

for col in target_enc_cols:
    train_eng[f'{col}_te'] = np.nan
    test_eng[f'{col}_te'] = np.nan
    
    # Clean OOF training map configuration to prevent future validation leakage
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(train_eng):
        stats = pd.DataFrame({'cat': train_eng[col].iloc[train_idx].fillna('missing'), 'target': y.iloc[train_idx]})
        agg = stats.groupby('cat')['target'].agg(['mean', 'count'])
        smoothed_map = (agg['mean'] * agg['count'] + global_mean * SMOOTH) / (agg['count'] + SMOOTH)
        
        train_eng.loc[val_idx, f'{col}_te'] = train_eng.loc[val_idx, col].fillna('missing').map(smoothed_map)
        
    # Test set mapping based strictly on complete train distributions
    stats_full = pd.DataFrame({'cat': train_eng[col].fillna('missing'), 'target': y})
    agg_full = stats_full.groupby('cat')['target'].agg(['mean', 'count'])
    full_smoothed_map = (agg_full['mean'] * agg_full['count'] + global_mean * SMOOTH) / (agg_full['count'] + SMOOTH)
    
    test_eng[f'{col}_te'] = test_eng[col].fillna('missing').map(full_smoothed_map)
    
    train_eng[f'{col}_te'] = train_eng[f'{col}_te'].fillna(global_mean)
    test_eng[f'{col}_te'] = test_eng[f'{col}_te'].fillna(global_mean)

features_updated = [col for col in train_eng.columns if col not in drop_cols]

# Prepare category objects for LightGBM
for col in features_updated:
    if train_eng[col].dtype == 'object':
        train_eng[col] = train_eng[col].astype(str).fillna('missing').astype('category')
        test_eng[col] = test_eng[col].astype(str).fillna('missing').astype('category')

X = train_eng[features_updated]
X_test = test_eng[features_updated]

# ==========================================
# 3. CHRONOLOGICAL VALIDATION SETUP
# ==========================================
sort_idx = train_eng['TransactionDate'].sort_values().index
X_sorted, y_sorted = X.iloc[sort_idx], y.iloc[sort_idx]

split_idx = int(len(X_sorted) * 0.8)
X_train, X_val = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
y_train, y_val = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]

# ==========================================
# 4. MULTI-SEED LIGHTGBM LOOP
# ==========================================
seeds = [42, 100, 2026]
test_preds_accumulator = np.zeros(len(X_test))
val_preds_accumulator = np.zeros(len(X_val))

# Tuned parameters specifically focused on higher structural depth configurations
base_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 4000,
    'learning_rate': 0.02,
    'num_leaves': 95,            # Higher capacity to match advanced engineered interactions
    'max_depth': 10,
    'subsample': 0.85,
    'colsample_bytree': 0.70,
    'reg_alpha': 0.1,
    'reg_lambda': 5.0,           # Higher L2 penalty limits category memorization
    'verbose': -1,
    'n_jobs': -1 
}

for seed in seeds:
    print(f"\n--- Training LightGBM Instance (Seed: {seed}) ---")
    current_params = base_params.copy()
    current_params['random_state'] = seed
    
    model = lgb.LGBMRegressor(**current_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
    )
    
    val_preds_accumulator += model.predict(X_val) / len(seeds)
    best_iters = model.best_iteration_ if model.best_iteration_ else 2000
    
    print(f"Retraining full production space for {int(best_iters * 1.1)} iterations...")
    full_params = current_params.copy()
    full_params['n_estimators'] = int(best_iters * 1.1)
    
    full_model = lgb.LGBMRegressor(**full_params)
    full_model.fit(X, y)
    
    test_preds_accumulator += full_model.predict(X_test) / len(seeds)

print(f"\n>>> Combined Ensembled Validation RMSLE: {root_mean_squared_error(y_val, val_preds_accumulator):.5f} <<<")

# ==========================================
# 5. SUBMISSION GENERATION
# ==========================================
final_test_preds_raw = np.expm1(test_preds_accumulator).clip(0, None)

submission = pd.DataFrame({
    'TransactionID': test_df['TransactionID'],
    'TargetValue': final_test_preds_raw
})
submission.to_csv('submission.csv', index=False)
print("\nSubmission output generated successfully on single tree model pipeline!")