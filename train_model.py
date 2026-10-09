import pandas as pd
import numpy as np
import joblib

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print("=" * 60)
print("EXPERIMENT 5 - CKD MODEL TRAINING")
print("=" * 60)

# ------------------------------------------------------------
# 1. LOAD OFFICIAL UCI DATASET
# ------------------------------------------------------------

print("\nDownloading CKD dataset from UCI...")

try:
    dataset = fetch_ucirepo(id=336)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    print("Dataset downloaded successfully!")

except Exception as e:
    print("\nERROR:", e)
    exit()


# ------------------------------------------------------------
# 2. CLEAN COLUMN NAMES
# ------------------------------------------------------------

X.columns = X.columns.str.strip().str.lower()

print("\nDataset shape:", X.shape)
print("\nAvailable features:")
print(X.columns.tolist())


# ------------------------------------------------------------
# 3. CLEAN TARGET
# ------------------------------------------------------------

if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

y = (
    y.astype(str)
    .str.strip()
    .str.lower()
)

y = y.replace({
    "ckd": 1,
    "notckd": 0,
    "not ckd": 0,
    "not-ckd": 0
})

y = pd.to_numeric(y, errors="coerce")

valid = y.notna()

X = X.loc[valid].copy()
y = y.loc[valid].astype(int)


# ------------------------------------------------------------
# 4. SELECT FEATURES
# ------------------------------------------------------------

features = [
    "age",
    "bp",
    "sg",
    "al",
    "bgr",
    "bu",
    "sc",
    "hemo",
    "htn",
    "dm"
]

missing = [f for f in features if f not in X.columns]

if missing:
    print("\nMissing features:", missing)
    exit()

X = X[features].copy()


# ------------------------------------------------------------
# 5. FEATURE TYPES
# ------------------------------------------------------------

numeric_features = [
    "age",
    "bp",
    "sg",
    "al",
    "bgr",
    "bu",
    "sc",
    "hemo"
]

categorical_features = [
    "htn",
    "dm"
]


# ------------------------------------------------------------
# 6. CLEAN NUMERICAL DATA
# ------------------------------------------------------------

for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")


# ------------------------------------------------------------
# 7. CLEAN CATEGORICAL DATA
# ------------------------------------------------------------

for col in categorical_features:
    X[col] = (
        X[col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

X = X.replace({
    "nan": np.nan,
    "none": np.nan,
    "?": np.nan,
    "": np.nan
})


# ------------------------------------------------------------
# 8. DISPLAY INFORMATION
# ------------------------------------------------------------

print("\nNumber of records:", len(X))
print("Number of features:", len(features))

print("\nTarget distribution:")
print(y.value_counts())


# ------------------------------------------------------------
# 9. TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# 10. PREPROCESSING
# ------------------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ------------------------------------------------------------
# 11. MODEL
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


# ------------------------------------------------------------
# 12. COMPLETE PIPELINE
# ------------------------------------------------------------

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# ------------------------------------------------------------
# 13. TRAIN
# ------------------------------------------------------------

print("\nTraining Random Forest model...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# ------------------------------------------------------------
# 14. PREDICTION
# ------------------------------------------------------------

y_pred = pipeline.predict(X_test)


# ------------------------------------------------------------
# 15. EVALUATION
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not CKD", "CKD"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ------------------------------------------------------------
# 16. SAVE MODEL
# ------------------------------------------------------------

joblib.dump(pipeline, "ckd_model.pkl")

print("\nSaved: ckd_model.pkl")


# ------------------------------------------------------------
# 17. SAVE MODEL INFORMATION
# ------------------------------------------------------------

model_info = {
    "model_name": "CKD Prediction Model",
    "model_version": "1.0",
    "algorithm": "Random Forest Classifier",
    "accuracy": round(float(accuracy), 4),
    "accuracy_percentage": round(float(accuracy * 100), 2),
    "number_of_features": len(features),
    "features": features,
    "classes": {
        "0": "Not CKD",
        "1": "CKD"
    }
}

joblib.dump(model_info, "model_info.pkl")

print("Saved: model_info.pkl")


# ------------------------------------------------------------
# 18. SAVE CLEAN DATASET
# ------------------------------------------------------------

clean_data = X.copy()
clean_data["class"] = y.values

clean_data.to_csv(
    "ckd_clean.csv",
    index=False
)

print("Saved: ckd_clean.csv")


# ------------------------------------------------------------
# 19. FINISH
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TASK 1 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFiles created:")
print("1. ckd_model.pkl")
print("2. model_info.pkl")
print("3. ckd_clean.csv")