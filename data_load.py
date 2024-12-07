import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_parquet(file_path):
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:  # Catch specific exceptions if needed
        print(f"Error loading parquet file: {e}")
        return None  # Return None to indicate failure
    return df

def replace_inf(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Flow Bytes/s', ' Flow Packets/s'], inplace=True)
    assert not df.isin([np.inf, -np.inf]).any().any(), "Infinity values still exist!"
    assert not df.isna().any().any(), "NaN values still exist!"
    return df

def map_labels(df,col_name):
    map_dict = dict()
    val = 0
    for label in df[col_name].unique():
        map_dict[label] = val
        val += 1
    df[col_name] = df[col_name].map(map_dict)
    return df, map_dict

def split_scale(df,target_col_name,feature_col_name):
    Y = df.loc[:,target_col_name] 
    X = df.loc[:,feature_col_name]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle = False, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)
    return X_train, X_test, Y_train, Y_test

def get_data(file_path, target_col_name, feature_col_name):
    df = load_parquet(file_path)
    if df is None:
        raise ValueError("Failed to load parquet file. Please check the file path or content.")
    
    df = replace_inf(df)
    df, mapping = map_labels(df, target_col_name)
    X_train, X_test, Y_train, Y_test = split_scale(df, target_col_name, feature_col_name)
    return X_train, X_test, Y_train, Y_test, mapping

# X_train, X_test, Y_train, Y_test = get_data('traffic_data.parquet')