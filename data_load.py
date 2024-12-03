import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_parquet(file_path):
    try:
        df = pd.read_parquet(file_path)
    except:
        print("File path not correct")
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

def get_data(file_path):
    df = replace_inf(load_parquet(file_path))
    df, mapping = map_labels(df,' Label')
    X_train, X_test, Y_train, Y_test = split_scale(df,' Label',[' Flow Duration', 'Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std', 'Bwd Packet Length Max', ' Bwd Packet Length Min', ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Fwd URG Flags', ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size', ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', ' Fwd Header Length.1', 'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min'])
    return X_train, X_test, Y_train, Y_test

# X_train, X_test, Y_train, Y_test = get_data('traffic_data.parquet')
