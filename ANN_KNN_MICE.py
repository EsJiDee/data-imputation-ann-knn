#-------------------SEED-----------------------
RANDOM_SEED = 123

import os
import random
import platform
import sys
import warnings
import logging
from sklearn.exceptions import ConvergenceWarning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def clear_screen():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
df = pd.read_csv(url, header=None, usecols=[0, 1, 2, 3], names=cols)
X_COL = "sepal_length"
TARGETS = ["sepal_width", "petal_length", "petal_width"]
all_cols = [X_COL] + TARGETS

'''
clear_screen()
print("Your code is running... Please Wait")
sys.stdout.flush()
'''

def induce_mcar(df, missing_rate, targets=TARGETS, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    data = df.copy()
    for col in targets:
        mask = rng.rand(len(data)) < missing_rate
        data.loc[mask, col] = np.nan
    return data

def induce_mar(df, missing_rate, targets=TARGETS, pivot_col="sepal_length", seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    data = df.copy()
    median = data[pivot_col].median()
    prob_given_condition = missing_rate / 0.5
    if prob_given_condition > 1.0:
        prob_given_condition = 1.0
    for col in targets:
        condition_mask = data[pivot_col] > median
        rand_mask = rng.rand(len(data)) < prob_given_condition
        
        mask = condition_mask & rand_mask
        data.loc[mask, col] = np.nan
    return data

def induce_mnar(df, missing_rate, targets=TARGETS, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    data = df.copy()
    for col in targets:
        median = data[col].median()
        condition_mask = data[col] > median
        condition_proportion = condition_mask.mean()
        if condition_proportion == 0:
            continue
        prob_given_condition = missing_rate / condition_proportion
        if prob_given_condition > 1.0:
            prob_given_condition = 1.0
        rand_mask = rng.rand(len(data)) < prob_given_condition
        mask = condition_mask & rand_mask
        data.loc[mask, col] = np.nan
    return data

def build_ann(input_dim=1, hidden_units=16, seed=RANDOM_SEED):
    tf.random.set_seed(seed)
    model = Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),  
        Dense(hidden_units, activation="relu"),   
        Dense(hidden_units, activation="relu"),
        Dense(len(TARGETS))
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def iterative_ann_imputer(train_df, max_iter=5, verbose=False):
    df = train_df.copy()
    scaler = StandardScaler()
    df[X_COL] = df[X_COL].astype(float)

    missing_row_idx = train_df[TARGETS].isna().any(axis=1)
    if not missing_row_idx.any():
        return df
    for t in TARGETS:
        mean_val = df[t].mean(skipna=True)
        df[t] = df[t].fillna(mean_val)

    for iteration in range(max_iter):
        X_scaled = scaler.fit_transform(df[[X_COL]])
        y = df[TARGETS].values
        model = build_ann(input_dim=1, hidden_units=16)
        model.fit(X_scaled, y, epochs=100, verbose=0)
        X_to_predict = scaler.transform(train_df.loc[missing_row_idx, [X_COL]])
        preds = model.predict(X_to_predict, verbose=0)
        df.loc[missing_row_idx, TARGETS] = preds
        if verbose:
            print(f"Iteration {iteration + 1}/{max_iter} completed.")
    return df

mechanisms = ["MCAR", "MAR", "MNAR"]
missing_rates = [0.30, 0.50, 0.70]
results = []

train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

for mech in mechanisms:
    for mr in missing_rates:
        if mech == "MCAR":
            train_miss_data = induce_mcar(train_df, mr, seed=RANDOM_SEED)
        elif mech == "MAR":
            train_miss_data = induce_mar(train_df, mr, pivot_col="sepal_length", seed=RANDOM_SEED)
        elif mech == "MNAR":
            train_miss_data = induce_mnar(train_df, mr, seed=RANDOM_SEED)

        df_ann_imputed = iterative_ann_imputer(train_miss_data, max_iter=5)
        missing_mask = train_miss_data[TARGETS].isna()
        true_values = train_df[TARGETS].values[missing_mask.values]
        imputed_values = df_ann_imputed[TARGETS].values[missing_mask.values]
        mse_ann = mean_squared_error(true_values, imputed_values)
        results.append({
            "Mechanism": mech,
            "Missing%": int(mr * 100),
            "Method": "ANN",
            "MSE": mse_ann
        })

        knn_imputer = KNNImputer(n_neighbors=5)
        df_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(train_miss_data[all_cols]), columns=all_cols)
        missing_mask = train_miss_data[TARGETS].isna()
        true_values = train_df[TARGETS].values[missing_mask.values]
        imputed_values = df_knn_imputed[TARGETS].values[missing_mask.values]
        mse_knn = mean_squared_error(true_values, imputed_values)
        results.append({
            "Mechanism": mech,
            "Missing%": int(mr * 100),
            "Method": "KNN",
            "MSE": mse_knn
        })

        mice_imputer = IterativeImputer(
            estimator=BayesianRidge(),
            initial_strategy="mean",
            max_iter=5,
            tol=0.01,
            random_state=RANDOM_SEED,
            verbose=0
        )
        df_mice_imputed = pd.DataFrame(mice_imputer.fit_transform(train_miss_data[all_cols]), columns=all_cols)
        missing_mask = train_miss_data[TARGETS].isna()
        true_values = train_df[TARGETS].values[missing_mask.values]
        imputed_values = df_mice_imputed[TARGETS].values[missing_mask.values]
        mse_mice = mean_squared_error(true_values, imputed_values)
        results.append({
            "Mechanism": mech,
            "Missing%": int(mr * 100),
            "Method": "MICE",
            "MSE": mse_mice
        })

summary_df = pd.DataFrame(results)
summary_df = summary_df[["Mechanism", "Missing%", "Method", "MSE"]]
summary_df = summary_df.sort_values(["Mechanism", "Missing%", "Method"]).reset_index(drop=True)

clear_screen()

print("Imputation Performance Comparison :-")
print(summary_df)
pivot = summary_df.pivot_table(index=["Mechanism", "Missing%"], columns="Method", values=["MSE"])
print("\nSummary:-")
print(pivot)