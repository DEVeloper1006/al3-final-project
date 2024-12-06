import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

# Define attack types for each level
level_1_labels = {
    'BENIGN': 0,
    'ATTACK': 1
}

group_labels = {
    'DoS/DDoS': ['DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'DDoS'],
    'Brute Force': ['FTP-Patator', 'SSH-Patator', 'Heartbleed', 'Bot'],
    'Reconnaissance': ['PortScan', 'Infiltration']
}

# Map the level 2 groups to numerical encoding
group_mapping = {
    'DoS/DDoS': 0,
    'Brute Force': 1,
    'Reconnaissance': 2
}

# Define selected features for training
selected_features = [
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
    'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 
    'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' act_data_pkt_fwd', 
    ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 
    'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min'
]

# Function to create hierarchical labels
def create_hierarchical_labels(data):
    """Create hierarchical labels for each level of classification"""
    
    # Level 1: Binary classification (BENIGN vs ATTACK)
    data['Level_1_Label'] = data[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # Level 2: Attack group classification
    def get_group_label(attack_type):
        if attack_type == 'BENIGN':
            return -1
        for group, attacks in group_labels.items():
            if attack_type in attacks:
                return group_mapping[group]
        return -1
    
    data['Level_2_Label'] = data[' Label'].apply(get_group_label)
    
    
    # Level 3: Specific attack type within group
    attack_type_mapping = {}
    for group, attacks in group_labels.items():
        for i, attack in enumerate(attacks):
            attack_type_mapping[attack] = i
    
    def get_specific_attack_label(row):
        if row[' Label'] == 'BENIGN':
            return -1
        return attack_type_mapping.get(row[' Label'], -1)
    
    data['Level_3_Label'] = data.apply(get_specific_attack_label, axis=1)
    
    return data

# Prepare data with a holdout set
def prepare_data_with_holdout():
    print("Loading and preprocessing data...")
    file_path = 'traffic_data.parquet'
    data = pd.read_parquet(file_path)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=['Flow Bytes/s', ' Flow Packets/s'], inplace=True)
    data = create_hierarchical_labels(data)

    balanced_samples = []
    max_samples_per_class = 2500
    for label in np.unique(data[' Label']):
        if label != 'BENIGN':
            class_data = data[data[' Label'] == label]
            if len(class_data) > max_samples_per_class:
                class_data = class_data.sample(max_samples_per_class, random_state=42)
            balanced_samples.append(class_data)
        else:
            class_data = data[data[' Label'] == label]
            if len(class_data) > max_samples_per_class:
                class_data = class_data.sample(25000, random_state=42)
            balanced_samples.append(class_data)
    balanced_data = pd.concat(balanced_samples)

    X = balanced_data[selected_features]
    y_level_1 = balanced_data['Level_1_Label']
    y_level_2 = balanced_data['Level_2_Label']
    y_level_3 = balanced_data['Level_3_Label']

    X_temp, X_holdout, y_l1_temp, y_l1_holdout, y_l2_temp, y_l2_holdout, y_l3_temp, y_l3_holdout = train_test_split(
        X, y_level_1, y_level_2, y_level_3,
        test_size=0.2, random_state=42, stratify=y_level_1
    )

    X_train, X_test, y_train_l1, y_test_l1, y_train_l2, y_test_l2, y_train_l3, y_test_l3 = train_test_split(
        X_temp, y_l1_temp, y_l2_temp, y_l3_temp,
        test_size=0.25, random_state=42, stratify=y_l1_temp
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_holdout_scaled = scaler.transform(X_holdout)

    return (X_train_scaled, X_test_scaled, X_holdout_scaled,
            y_train_l1, y_test_l1, y_l1_holdout,
            y_train_l2, y_test_l2, y_l2_holdout,
            y_train_l3, y_test_l3, y_l3_holdout)

# Train models for each level
def train_and_evaluate(X_train, y_train_l1, y_train_l2, y_train_l3, X_test, y_test_l1, y_test_l2, y_test_l3):
    # Level 1 model
    level_1_model = SVC(kernel='rbf', random_state=42, C=2)
    level_1_model.fit(X_train, y_train_l1)
    print("\nLevel 1 Evaluation:")
    print(classification_report(y_test_l1, level_1_model.predict(X_test), zero_division=0))

    # Level 2 model (on ATTACK samples)
    attack_mask_train = y_train_l1 == 1
    attack_mask_test = y_test_l1 == 1
    level_2_model = SVC(kernel='linear', random_state=42, C=0.5)
    level_2_model.fit(X_train[attack_mask_train & (y_train_l2 != -1)], y_train_l2[attack_mask_train & (y_train_l2 != -1)])
    print("\nLevel 2 Evaluation:")
    print(classification_report(y_test_l2[attack_mask_test & (y_test_l2 != -1)], level_2_model.predict(X_test[attack_mask_test & (y_test_l2 != -1)]), zero_division=0))

    # Level 3 models for each group
    level_3_models = {}
    for group_id in range(3):
        group_mask_train = (y_train_l2 == group_id) & attack_mask_train
        group_mask_test = (y_test_l2 == group_id) & attack_mask_test
        if np.any(group_mask_train):
            # Apply SMOTE to balance the training data for this group
            smote = SMOTE(random_state=42)
            X_group_train = X_train[group_mask_train & (y_train_l3 != -1)]
            y_group_train = y_train_l3[group_mask_train & (y_train_l3 != -1)]
            X_group_train_smote, y_group_train_smote = smote.fit_resample(X_group_train, y_group_train)

            group_model = SVC(kernel='linear', random_state=42, C=0.01)
            group_model.fit(X_group_train_smote, y_group_train_smote)
            level_3_models[group_id] = group_model

            if np.any(group_mask_test):
                print(f"\nLevel 3 Evaluation for Group {group_id}:")
                print(classification_report(y_test_l3[group_mask_test & (y_test_l3 != -1)], group_model.predict(X_test[group_mask_test & (y_test_l3 != -1)]), zero_division=0))
        else:
            level_3_models[group_id] = None

    return level_1_model, level_2_model, level_3_models

# Predict on the holdout set
def predict_pipeline(models, X_holdout_scaled):
    level_1_model, level_2_model, level_3_models = models
    pred_l1 = level_1_model.predict(X_holdout_scaled)
    attack_mask = pred_l1 == 1
    pred_l2 = np.full(pred_l1.shape, -1)
    if np.any(attack_mask):
        pred_l2[attack_mask] = level_2_model.predict(X_holdout_scaled[attack_mask])
    pred_l3 = np.full(pred_l1.shape, -1)
    for group_id, group_model in level_3_models.items():
        if group_model is not None:
            group_mask = (pred_l2 == group_id) & attack_mask
            if np.any(group_mask):
                pred_l3[group_mask] = group_model.predict(X_holdout_scaled[group_mask])
    return pred_l1, pred_l2, pred_l3

def evaluate_holdout(models, X_holdout_scaled, y_l1_holdout, y_l2_holdout, y_l3_holdout):
    pred_l1, pred_l2, pred_l3 = predict_pipeline(models, X_holdout_scaled)
    
    level_2_names = {
        0: 'DoS/DDoS',
        1: 'Brute Force',
        2: 'Reconnaissance'
    }
    
    level_3_names = {
        0: {  # DoS/DDoS attacks
            0: 'DoS GoldenEye',
            1: 'DoS Hulk',
            2: 'DoS Slowhttptest',
            3: 'DoS slowloris',
            4: 'DDoS'
        },
        1: {  # Brute Force attacks
            0: 'FTP-Patator',
            1: 'SSH-Patator',
            2: 'Heartbleed',
            3: 'Bot'
        },
        2: {  # Reconnaissance attacks
            0: 'PortScan',
            1: 'Infiltration'
        }
    }
    
    print("\nHoldout Evaluation - Level 1:")
    print(classification_report(y_l1_holdout, pred_l1, 
                              target_names=['BENIGN', 'ATTACK'],
                              zero_division=0))
    
    print("\nHoldout Evaluation - Level 2:")
    attack_mask = y_l1_holdout == 1
    valid_l2_mask = (y_l2_holdout != -1) & attack_mask
    print(classification_report(y_l2_holdout[valid_l2_mask], 
                              pred_l2[valid_l2_mask], 
                              target_names=[level_2_names[i] for i in range(3)],
                              zero_division=0,
                              labels=[0, 1, 2]))
    
    print("\nHoldout Evaluation - Level 3:")
    for group_id in range(3):
        group_mask = (y_l2_holdout == group_id) & attack_mask
        valid_l3_mask = (y_l3_holdout != -1) & group_mask
        if np.sum(valid_l3_mask) > 0:
            valid_labels = sorted(list(set(y_l3_holdout[valid_l3_mask])))
            print(f"\nGroup: {level_2_names[group_id]}")
            print(classification_report(y_l3_holdout[valid_l3_mask], 
                                     pred_l3[valid_l3_mask], 
                                     target_names=[level_3_names[group_id][i] for i in valid_labels],
                                     zero_division=0,
                                     labels=valid_labels))

def prepare_data_for_cv():
    print("Loading and preprocessing data...")
    file_path = 'traffic_data.parquet'
    data = pd.read_parquet(file_path)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=['Flow Bytes/s', ' Flow Packets/s'], inplace=True)
    data = create_hierarchical_labels(data)

    # Balance the dataset
    balanced_samples = []
    max_samples_per_class = 2500
    for label in np.unique(data[' Label']):
        class_data = data[data[' Label'] == label]
        if label != 'BENIGN':
            if len(class_data) > max_samples_per_class:
                class_data = class_data.sample(max_samples_per_class, random_state=42)
        else:
            if len(class_data) > max_samples_per_class:
                class_data = class_data.sample(25000, random_state=42)
        balanced_samples.append(class_data)
    
    balanced_data = pd.concat(balanced_samples)
    
    X = balanced_data[selected_features]
    y_level_1 = balanced_data['Level_1_Label']
    y_level_2 = balanced_data['Level_2_Label']
    y_level_3 = balanced_data['Level_3_Label']
    
    return X, y_level_1, y_level_2, y_level_3

def cross_validate_hierarchical(X, y_level_1, y_level_2, y_level_3, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store metrics for each fold
    metrics = {
        'level_1': [],
        'level_2': [],
        'level_3': {0: [], 1: [], 2: []}
    }
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_level_1)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_l1, y_test_l1 = y_level_1.iloc[train_idx], y_level_1.iloc[test_idx]
        y_train_l2, y_test_l2 = y_level_2.iloc[train_idx], y_level_2.iloc[test_idx]
        y_train_l3, y_test_l3 = y_level_3.iloc[train_idx], y_level_3.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Level 1: Binary classification
        level_1_model = SVC(kernel='rbf', random_state=42, C=1)
        level_1_model.fit(X_train_scaled, y_train_l1)
        l1_preds = level_1_model.predict(X_test_scaled)
        l1_report = classification_report(y_test_l1, l1_preds, output_dict=True)
        metrics['level_1'].append(l1_report)
        
        # Level 2: Attack group classification
        attack_mask_train = y_train_l1 == 1
        attack_mask_test = y_test_l1 == 1
        
        level_2_model = SVC(kernel='linear', random_state=42, C=0.5)
        valid_l2_mask_train = attack_mask_train & (y_train_l2 != -1)
        valid_l2_mask_test = attack_mask_test & (y_test_l2 != -1)
        
        if np.any(valid_l2_mask_train):
            level_2_model.fit(X_train_scaled[valid_l2_mask_train], 
                            y_train_l2[valid_l2_mask_train])
            l2_preds = level_2_model.predict(X_test_scaled[valid_l2_mask_test])
            l2_report = classification_report(y_test_l2[valid_l2_mask_test], 
                                           l2_preds, output_dict=True)
            metrics['level_2'].append(l2_report)
        
        # Level 3: Specific attack classification
        for group_id in range(3):
            group_mask_train = (y_train_l2 == group_id) & attack_mask_train
            group_mask_test = (y_test_l2 == group_id) & attack_mask_test
            
            if np.any(group_mask_train):
                X_group_train = X_train_scaled[group_mask_train & (y_train_l3 != -1)]
                y_group_train = y_train_l3[group_mask_train & (y_train_l3 != -1)]
                
                # Apply SMOTE
                smote = SMOTE(random_state=42)
                X_group_train_smote, y_group_train_smote = smote.fit_resample(
                    X_group_train, y_group_train)
                
                group_model = SVC(kernel='linear', random_state=42, C=0.01)
                group_model.fit(X_group_train_smote, y_group_train_smote)
                
                valid_l3_mask_test = group_mask_test & (y_test_l3 != -1)
                if np.any(valid_l3_mask_test):
                    l3_preds = group_model.predict(X_test_scaled[valid_l3_mask_test])
                    l3_report = classification_report(y_test_l3[valid_l3_mask_test], 
                                                   l3_preds, output_dict=True)
                    metrics['level_3'][group_id].append(l3_report)
    
    return metrics

def print_cv_results(metrics):
    # Level 1 results
    l1_accuracy = np.mean([m['accuracy'] for m in metrics['level_1']])
    l1_std = np.std([m['accuracy'] for m in metrics['level_1']])
    print(f"\nLevel 1 CV Results:")
    print(f"Average Accuracy: {l1_accuracy:.3f} ± {l1_std:.3f}")
    
    # Level 2 results
    if metrics['level_2']:
        l2_accuracy = np.mean([m['accuracy'] for m in metrics['level_2']])
        l2_std = np.std([m['accuracy'] for m in metrics['level_2']])
        print(f"\nLevel 2 CV Results:")
        print(f"Average Accuracy: {l2_accuracy:.3f} ± {l2_std:.3f}")
    
    # Level 3 results
    print("\nLevel 3 CV Results by Group:")
    for group_id in range(3):
        if metrics['level_3'][group_id]:
            l3_accuracy = np.mean([m['accuracy'] for m in metrics['level_3'][group_id]])
            l3_std = np.std([m['accuracy'] for m in metrics['level_3'][group_id]])
            print(f"Group {group_id} Average Accuracy: {l3_accuracy:.3f} ± {l3_std:.3f}")

if __name__ == "__main__":
    X, y_level_1, y_level_2, y_level_3 = prepare_data_for_cv()
    metrics = cross_validate_hierarchical(X, y_level_1, y_level_2, y_level_3)
    print_cv_results(metrics)
    (X_train_scaled, X_test_scaled, X_holdout_scaled,
     y_train_l1, y_test_l1, y_l1_holdout,
     y_train_l2, y_test_l2, y_l2_holdout,
     y_train_l3, y_test_l3, y_l3_holdout) = prepare_data_with_holdout()

    models = train_and_evaluate(X_train_scaled, y_train_l1, y_train_l2, y_train_l3,
                                X_test_scaled, y_test_l1, y_test_l2, y_test_l3)

    print("\nEvaluating on holdout set...")
    evaluate_holdout(models, X_holdout_scaled, y_l1_holdout, y_l2_holdout, y_l3_holdout)