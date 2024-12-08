import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

# def process_files_to_parquet():
#     path = "TrafficLabelling/"
#     data_frames = []

#     file_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
#     for file_path in file_paths:
#         df_added = pd.read_csv(file_path)
#         data_frames.append(df_added)
#     if data_frames:
#         final_df = pd.concat(data_frames,ignore_index=True)
#         final_df.to_parquet('traffic_data_1.parquet',index=False)
        
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

def get_data(file_path, target_col_name, feature_col_name):
    df = load_parquet(file_path)
    if df is None:
        raise ValueError("Failed to load parquet file. Please check the file path or content.")
    
    df = replace_inf(df)
    
    Y = df.loc[:, target_col_name]
    X = df.loc[:, feature_col_name]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    #X_train, X_test, Y_train, Y_test = split_scale(df, target_col_name, feature_col_name)
    return X_train, X_test, Y_train, Y_test
    

features_list = [
        ' Flow Duration', 'Total Length of Fwd Packets', ' Total Length of Bwd Packets', 
        ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean', 
        ' Fwd Packet Length Std', 'Bwd Packet Length Max', ' Bwd Packet Length Min', 
        ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s', 
        ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', 
        ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', 
        ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', 
        ' Fwd URG Flags', ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s', 
        ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', 
        ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count', 
        ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count', 
        ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size', 
        ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', ' Fwd Header Length.1', 
        'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', 
        ' Subflow Bwd Bytes', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', 
        ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean', ' Active Std', 
        ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min'
]

X_train, X_test, Y_train, Y_test = get_data('traffic_data.parquet', ' Label', features_list) 

Y_train = Y_train.to_numpy().reshape(-1, 1)
Y_test = Y_test.to_numpy().reshape(-1, 1)

# Combine X_train and Y_train with proper column names
train = pd.concat([X_train, pd.DataFrame(Y_train, columns=[' Label'], index=X_train.index)], axis=1)
test = pd.concat([X_test, pd.DataFrame(Y_test, columns=[' Label'], index=X_test.index)], axis=1)

train.to_parquet('train_features.parquet', index=False)
test.to_parquet('test_features.parquet', index=False)
