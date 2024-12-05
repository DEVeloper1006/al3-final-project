import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

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

# Selected features (your existing features list)
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

def prepare_data():
    print("Loading and preprocessing data...")
    # Load the data
    file_path = 'traffic_data.parquet'
    data = pd.read_parquet(file_path)
    
    # Handle infinities and NaN values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=['Flow Bytes/s', ' Flow Packets/s'], inplace=True)
    
    # Create hierarchical labels
    data = create_hierarchical_labels(data)
    
    # Balance classes (1000 samples per original attack type)
    balanced_samples = []
    max_samples_per_class = 5000
    
    for label in np.unique(data[' Label']):
        if label != 'BENIGN':

            class_data = data[data[' Label'] == label]
            if len(class_data) > max_samples_per_class:
                class_data = class_data.sample(max_samples_per_class, random_state=42)
            balanced_samples.append(class_data)
        else:
            
            class_data = data[data[' Label'] == label]
            if len(class_data) > max_samples_per_class:
                class_data = class_data.sample(42000, random_state=42)
            balanced_samples.append(class_data)
            
    
    balanced_data = pd.concat(balanced_samples)
    
    # Prepare features and labels
    X = balanced_data[selected_features]
    y_level_1 = balanced_data['Level_1_Label']
    
    y_level_2 = balanced_data['Level_2_Label']
    
    y_level_3 = balanced_data['Level_3_Label']
    
    
    # Train-test split
    X_train, X_test, y_train_l1, y_test_l1, y_train_l2, y_test_l2, y_train_l3, y_test_l3 = train_test_split(
        X, y_level_1, y_level_2, y_level_3,
        test_size=0.2, random_state=42, stratify=y_level_1
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_test_scaled, 
            y_train_l1, y_test_l1,
            y_train_l2, y_test_l2,
            y_train_l3, y_test_l3)

svm_level_3_models = {}
def get_attack_names():
    """Create mappings between attack names and their group/specific indices"""
    attack_names_by_group = {}
    for group_id, (group_name, attacks) in enumerate(group_labels.items()):
        attack_names_by_group[group_id] = {
            idx: attack_name 
            for idx, attack_name in enumerate(attacks)
        }
    return attack_names_by_group

def train_and_evaluate():
    # Prepare data
    (X_train_scaled, X_test_scaled,
     y_train_l1, y_test_l1,
     y_train_l2, y_test_l2,
     y_train_l3, y_test_l3) = prepare_data()
    
    print("Training Level 1 (BENIGN vs ATTACK)...")
    # Level 1: Binary classification
    svm_level_1 = SVC(kernel='rbf', class_weight='balanced', probability=True)
    svm_level_1.fit(X_train_scaled, y_train_l1)
    y_pred_l1 = svm_level_1.predict(X_test_scaled)
    
    print("Training Level 2 (Attack Groups)...")
    # Level 2: Attack group classification (only for attack samples)
    attack_mask_train = y_train_l1 == 1
    attack_mask_test = y_test_l1 == 1
    
    X_train_attack = X_train_scaled[attack_mask_train]
    y_train_l2_attack = y_train_l2[attack_mask_train]
    
    svm_level_2 = SVC(kernel='rbf', class_weight='balanced', probability=True)
    svm_level_2.fit(X_train_attack, y_train_l2_attack)
    
    # For evaluation, we'll use the true attack samples
    X_test_attack = X_test_scaled[attack_mask_test]
    y_test_l2_attack = y_test_l2[attack_mask_test]
    y_pred_l2_attack = svm_level_2.predict(X_test_attack)
    
    print("Training Level 3 (Specific Attacks)...")
    # Get attack names mapping
    attack_names_by_group = get_attack_names()
    
    # Level 3: Specific attack classification (separate model for each group)
    for group_id in range(3):  # 3 attack groups
        group_name = [name for name, id in group_mapping.items() if id == group_id][0]
        print(f"\nTraining and evaluating attacks in group: {group_name}")
        
        # Train data for this group
        group_mask_train = (y_train_l2 == group_id) & attack_mask_train
        X_train_group = X_train_scaled[group_mask_train]
        y_train_l3_group = y_train_l3[group_mask_train]
        
        print(f"Group Id {group_id}, Distribution : {Counter(y_train_l3_group)}")
        if len(X_train_group) > 0:
            svm_level_3 = SVC(kernel='rbf', class_weight='balanced', probability=True)
            svm_level_3.fit(X_train_group, y_train_l3_group)
            svm_level_3_models[group_id] = svm_level_3
            
            # Test data for this group
            group_mask_test = (y_test_l2 == group_id) & attack_mask_test
            X_test_group = X_test_scaled[group_mask_test]
            y_test_l3_group = y_test_l3[group_mask_test]
            
            if len(X_test_group) > 0:
                y_pred_l3_group = svm_level_3.predict(X_test_group)
                
                # Create attack name mappings for the classification report
                attack_names = attack_names_by_group[group_id]
                target_names = [attack_names[i] for i in sorted(set(y_test_l3_group))]
                
                print(f"\nClassification Report for {group_name} attacks:")
                print(classification_report(
                    y_test_l3_group, 
                    y_pred_l3_group,
                    target_names=target_names
                ))
                
                
            else:
                print(f"No test samples for group {group_name}")
        else:
            print(f"No training samples for group {group_name}")
    
    # Print overall reports
    print("\nLevel 1 Classification Report:")
    print(classification_report(y_test_l1, y_pred_l1))
    
    print("\nLevel 2 Classification Report (Attack samples only):")
    group_names = [name for name, _ in group_mapping.items()]
    print(classification_report(y_test_l2_attack, y_pred_l2_attack, target_names=group_names))

if __name__ == "__main__":
    train_and_evaluate()