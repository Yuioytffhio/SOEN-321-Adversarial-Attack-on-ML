import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------------
# LOAD TRAIN & TEST DATA
# -----------------------------------
#train_file = "data/KDDTrain+.txt"
train_file = "data/KDDTrain+_balanced_to_poisoned_attackhiding 5%.txt"
test_file = "data/KDDTest+_balanced.txt"

# 41 features + 1 label
df_train = pd.read_csv(train_file, header=None)
df_test = pd.read_csv(test_file, header=None)

label_col = 41 

# -----------------------------------
# FEATURES & LABELS
# -----------------------------------
X_train = df_train.drop(columns=[label_col])
y_train = df_train[label_col].apply(lambda x: 0 if x == "normal" else 1)

X_test = df_test.drop(columns=[label_col])
y_test = df_test[label_col].apply(lambda x: 0 if x == "normal" else 1)

# -----------------------------------
# IDENTIFY CATEGORICAL & NUMERIC COLS
# -----------------------------------
categorical_cols = [1, 2, 3, 6, 11, 20, 21]
numeric_cols = sorted([i for i in range(X_train.shape[1]) if i not in categorical_cols])

# -----------------------------------
# PREPROCESSING PIPELINE
# -----------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("numerical", StandardScaler(), numeric_cols)
    ]
)

# Fit on train and transform both train & test
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert to dense if sparse
if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()
    X_test_processed = X_test_processed.toarray()

# -----------------------------------
# BUILD NEURAL NETWORK
# -----------------------------------
model = models.Sequential([
    layers.Input(shape=(X_train_processed.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------------
# TRAIN MODEL
# -----------------------------------
history = model.fit(
    X_train_processed, y_train,
    epochs=15,
    batch_size=1024,
    validation_split=0.1,  
    verbose=1
)

model.save("clean model/neural_model.keras")
print("\nModel saved as neural_model.keras")

# -----------------------------------
# EVALUATE ON OFFICIAL TEST SET
# -----------------------------------
loss, acc = model.evaluate(X_test_processed, y_test)
print("\nTest Accuracy on KDDTest+: {:.4f}".format(acc))

# -----------------------------------
# CONFUSION MATRIX & CLASSIFICATION REPORT
# -----------------------------------
y_pred_probs = model.predict(X_test_processed)
y_pred = (y_pred_probs > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

report = classification_report(y_test, y_pred, target_names=["normal", "attack"])
print("\nClassification Report:")
print(report)
