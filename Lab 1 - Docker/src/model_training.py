import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('data/penguins_size.csv')
    
    # Replace '.' with NaN in sex column
    df['sex'] = df['sex'].replace('.', pd.NA)
    
    # Drop rows with any missing values
    df_clean = df.dropna()
    
    # Encode categorical features
    le_island = LabelEncoder()
    df_clean['island'] = le_island.fit_transform(df_clean['island'])
    
    le_sex = LabelEncoder()
    df_clean['sex'] = le_sex.fit_transform(df_clean['sex'])
    
    # Separate features and target
    X = df_clean.drop('species', axis=1)
    y = df_clean['species']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model trained successfully!")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"Training set accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test set accuracy: {accuracy:.4f}")

    # Save model
    joblib.dump(model, 'penguin_model.pkl')
    print("\nModel saved as penguin_model.pkl")

