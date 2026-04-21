import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Constants
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"
DATA_FILE = "housing.csv"

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])
    return full_pipeline

# 1. DATA PREPARATION (Should happen regardless of training/inference status if files are missing)
if not os.path.exists("input.csv"):
    housing = pd.read_csv(DATA_FILE)
    housing['income_cat'] = pd.cut(housing["median_income"], 
                                   bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], 
                                   labels=[1, 2, 3, 4, 5])
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
        strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)
    
    # Save the test set as our "new unseen data" for inference later
    strat_test_set.to_csv("input.csv", index=False)
    strat_train_set.to_csv("train_internal.csv", index=False)

# 2. TRAINING PHASE
if not os.path.exists(MODEL_FILE):
    train_data = pd.read_csv("train_internal.csv")
    housing_labels = train_data["median_house_value"].copy()
    housing_features = train_data.drop("median_house_value", axis=1)

    # Define columns explicitly
    num_attribs = list(housing_features.drop("ocean_proximity", axis=1))
    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model trained and saved.")

# 3. INFERENCE PHASE
else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    # Load new data
    input_data = pd.read_csv("input.csv")
    
    # IMPORTANT: Remove the target column if it exists in the input data
    if "median_house_value" in input_data.columns:
        features = input_data.drop("median_house_value", axis=1)
    else:
        features = input_data

    transformed_input = pipeline.transform(features)
    predictions = model.predict(transformed_input)
    
    # Add predictions to the original dataframe
    input_data["predicted_house_value"] = predictions
    input_data.to_csv("output.csv", index=False)
    print("Inference complete. Results saved to output.csv")