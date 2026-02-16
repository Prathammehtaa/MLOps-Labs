import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

# File paths
DATA_PATH = '/opt/airflow/dags/data/work_from_home_burnout_dataset.csv'
TEMP_DIR = '/opt/airflow/dags/temp/'

def load_data():
    """Load the burnout dataset"""
    df = pd.read_csv(DATA_PATH)
    df.to_csv(f'{TEMP_DIR}raw_data.csv', index=False)
    print(f"Loaded {len(df)} rows")
    return "Data loaded"

def preprocess_data():
    """Preprocess data and split into train/test"""
    df = pd.read_csv(f'{TEMP_DIR}raw_data.csv')
    
    # Create binary target
    df['burnout_binary'] = df['burnout_risk'].apply(lambda x: 0 if x == 'Low' else 1)
    df = df.drop(['burnout_risk', 'user_id'], axis=1)
    
    # Encode day_type
    df['day_type'] = df['day_type'].map({'Weekday': 0, 'Weekend': 1})
    
    # Split features and target
    X = df.drop('burnout_binary', axis=1)
    y = df['burnout_binary']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Save to files
    X_train.to_csv(f'{TEMP_DIR}X_train.csv', index=False)
    X_test.to_csv(f'{TEMP_DIR}X_test.csv', index=False)
    y_train.to_csv(f'{TEMP_DIR}y_train.csv', index=False, header=True)
    y_test.to_csv(f'{TEMP_DIR}y_test.csv', index=False, header=True)
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return "Preprocessing complete"

def train_random_forest():
    """Train Random Forest"""
    X_train = pd.read_csv(f'{TEMP_DIR}X_train.csv')
    X_test = pd.read_csv(f'{TEMP_DIR}X_test.csv')
    y_train = pd.read_csv(f'{TEMP_DIR}y_train.csv').values.ravel()
    y_test = pd.read_csv(f'{TEMP_DIR}y_test.csv').values.ravel()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Save model and scores
    with open(f'{TEMP_DIR}rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    results = {'name': 'RandomForest', 'accuracy': accuracy, 'f1': f1}
    with open(f'{TEMP_DIR}rf_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"RF - Acc: {accuracy:.4f}, F1: {f1:.4f}")
    return "RF trained"

def train_xgboost():
    """Train XGBoost"""
    X_train = pd.read_csv(f'{TEMP_DIR}X_train.csv')
    X_test = pd.read_csv(f'{TEMP_DIR}X_test.csv')
    y_train = pd.read_csv(f'{TEMP_DIR}y_train.csv').values.ravel()
    y_test = pd.read_csv(f'{TEMP_DIR}y_test.csv').values.ravel()
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    with open(f'{TEMP_DIR}xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    results = {'name': 'XGBoost', 'accuracy': accuracy, 'f1': f1}
    with open(f'{TEMP_DIR}xgb_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"XGB - Acc: {accuracy:.4f}, F1: {f1:.4f}")
    return "XGB trained"

def train_logistic_regression():
    """Train Logistic Regression"""
    X_train = pd.read_csv(f'{TEMP_DIR}X_train.csv')
    X_test = pd.read_csv(f'{TEMP_DIR}X_test.csv')
    y_train = pd.read_csv(f'{TEMP_DIR}y_train.csv').values.ravel()
    y_test = pd.read_csv(f'{TEMP_DIR}y_test.csv').values.ravel()
    
    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    with open(f'{TEMP_DIR}lr_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    results = {'name': 'LogisticRegression', 'accuracy': accuracy, 'f1': f1}
    with open(f'{TEMP_DIR}lr_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"LR - Acc: {accuracy:.4f}, F1: {f1:.4f}")
    return "LR trained"

def compare_models():
    """Compare all models"""
    with open(f'{TEMP_DIR}rf_results.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open(f'{TEMP_DIR}xgb_results.pkl', 'rb') as f:
        xgb = pickle.load(f)
    with open(f'{TEMP_DIR}lr_results.pkl', 'rb') as f:
        lr = pickle.load(f)
    
    best = max([rf, xgb, lr], key=lambda x: x['f1'])
    
    print(f"Best model: {best['name']} with F1: {best['f1']:.4f}")
    
    with open(f'{TEMP_DIR}best_model_name.txt', 'w') as f:
        f.write(best['name'])
    with open(f'{TEMP_DIR}best_f1.txt', 'w') as f:
        f.write(str(best['f1']))
    
    return "Comparison complete"

def check_accuracy_threshold():
    """Check if best model passes threshold"""
    with open(f'{TEMP_DIR}best_f1.txt', 'r') as f:
        f1 = float(f.read())
    
    threshold = 0.65
    passed = f1 >= threshold
    
    print(f"F1: {f1:.4f}, Threshold: {threshold}, Passed: {passed}")
    
    with open(f'{TEMP_DIR}threshold_passed.txt', 'w') as f:
        f.write(str(passed))
    
    return "Threshold checked"

def save_model():
    """Save best model if threshold passed"""
    with open(f'{TEMP_DIR}threshold_passed.txt', 'r') as f:
        passed = f.read() == 'True'
    
    if not passed:
        print("Model didn't pass threshold. Not saving.")
        return "Model not saved"
    
    with open(f'{TEMP_DIR}best_model_name.txt', 'r') as f:
        model_name = f.read().strip()
    
    # Map model names to file names
    model_mapping = {
        'RandomForest': 'rf',
        'XGBoost': 'xgb',
        'LogisticRegression': 'lr'
    }
    
    model_prefix = model_mapping[model_name]
    model_file = f'{TEMP_DIR}{model_prefix}_model.pkl'
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Save to final location
    final_path = f'/opt/airflow/dags/{model_name}_final.pkl'
    with open(final_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved: {final_path}")
    return "Model saved"