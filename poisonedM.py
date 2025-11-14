import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler
import sys
import os

# Columns and constants

COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']
NUMERIC_COLS = list(set(COLUMNS[:-2]) - set(CATEGORICAL_COLS))

POISONED_MODEL_FILE = "kdd_detection_model_poisoned.keras"


# Load data 

def load_data(train_path, test_path):
    """Load NSL-KDD train and test sets (without difficulty column)."""
    try:
        train_df = pd.read_csv(
            train_path,
            names=COLUMNS[:-1],   # 42 cols: 41 features + label
            usecols=range(42)
        )
        test_df = pd.read_csv(
            test_path,
            names=COLUMNS[:-1],
            usecols=range(42)
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease download 'KDDTrain+.txt' and 'KDDTest+.txt' from:")
        print("https://www.kaggle.com/datasets/hassan06/nslkdd?resource=download")
        print("and place them in the 'data' folder.")
        sys.exit(1)

    return train_df, test_df


# -Data poisoning

def poison_train_labels(train_df, poison_frac=0.10, random_state=42):
    """
    Label-flip poisoning:
    - Take a fraction of attack samples (label != 'normal')
    - Relabel them as 'normal'.
    """
    poisoned = train_df.copy()
    rng = np.random.default_rng(random_state)

    attack_mask = poisoned['label'] != 'normal'
    attack_indices = poisoned[attack_mask].index.to_numpy()

    n_attack = len(attack_indices)
    n_poison = int(poison_frac * n_attack)

    if n_poison == 0:
        print("Warning: poison_frac too small, no samples poisoned.")
        return poisoned

    poisoned_indices = rng.choice(attack_indices, size=n_poison, replace=False)
    poisoned.loc[poisoned_indices, 'label'] = 'normal'

    print(f"Poisoning: flipped {n_poison} attack samples "
          f"({poison_frac*100:.1f}% of attack rows) to 'normal'.")
    return poisoned


# Preprocessing 

def preprocess(train_df, test_df):
    """
    One-hot encode categoricals, scale numeric features.
    Returns X_train, X_test, y_train, y_test.
    """
    print("Starting preprocessing...")

    # Binary labels: 0 = normal, 1 = attack
    y_train = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1).values
    y_test = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1).values

    X_train = train_df.drop('label', axis=1)
    X_test = test_df.drop('label', axis=1)

    # One-hot encode categoricals on combined data for consistency
    combined = pd.concat([X_train, X_test], axis=0)
    combined_enc = pd.get_dummies(combined, columns=CATEGORICAL_COLS, dtype=float)

    X_train_enc = combined_enc.iloc[:len(X_train)]
    X_test_enc = combined_enc.iloc[len(X_train):]

    print(f"Features after one-hot encoding: {X_train_enc.shape[1]}")

    # Scale numeric features
    scaler = StandardScaler()
    scaler.fit(X_train_enc[NUMERIC_COLS])

    X_train_enc[NUMERIC_COLS] = scaler.transform(X_train_enc[NUMERIC_COLS])
    X_test_enc[NUMERIC_COLS] = scaler.transform(X_test_enc[NUMERIC_COLS])

    print("Numeric features scaled.")
    return X_train_enc.values, X_test_enc.values, y_train, y_test


#DNN model 

def build_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # binary: normal vs attack
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Main: poisoned model only 

def main():
    TRAIN_FILE = "data/KDDTrain+.txt"
    TEST_FILE  = "data/KDDTest+.txt"

    # Load clean train + test
    train_df, test_df = load_data(TRAIN_FILE, TEST_FILE)

    # Create POISONED train set
    poisoned_train_df = poison_train_labels(train_df, poison_frac=0.10, random_state=42)

    # Preprocess poisoned train + clean test
    X_train_pois, X_test, y_train_pois, y_test = preprocess(poisoned_train_df, test_df)

    input_shape = X_train_pois.shape[1]

    # Build and train poisoned model
    model = build_model(input_shape)
    model.summary()

    print("\n--- Training POISONED model ---")
    history = model.fit(
        X_train_pois, y_train_pois,
        epochs=20,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    # Save poisoned model (optional, but matches project style)
    model.save(POISONED_MODEL_FILE)
    print(f"\nPoisoned model saved to '{POISONED_MODEL_FILE}'.")

    # Evaluate on CLEAN test set
    print("\n--- Evaluating POISONED model on CLEAN test data ---")
    results = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nPoisoned Model - Test Loss: {results[0]:.4f}")
    print(f"Poisoned Model - Test Accuracy: {results[1]*100:.2f}%")

    #a few predictions vs true labels
    N = 20
    print(f"\n--- Sample predictions on first {N} clean test rows ---")
    preds = (model.predict(X_test[:N]) > 0.5).astype(int).ravel()
    for i in range(N):
        true_label = y_test[i]
        print(f"Sample {i+1:02d} | True: {true_label} | Pred (poisoned model): {preds[i]}")


if __name__ == "__main__":
    main()