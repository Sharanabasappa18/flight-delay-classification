import os
import time
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.data_loader import get_train_test_data
from src.preprocessing import fit_and_save_preprocessor, transform_data

MODELS_DIR = "models"


def get_models():
    """
    Initializes the three target classification algorithms with tuned parameters.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            C=1.0,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=16,
            min_samples_split=6,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )
    }
    return models


def train_all_models(
    data_path: str = "data/raw/flights.csv",
    save_models: bool = True
):
    """
    Loads data, trains Logistic Regression, Random Forest, and Gradient Boosting,
    and returns trained models alongside preprocessed test sets.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Splitting dataset into train & test splits...")
    X_train, X_test, y_train, y_test = get_train_test_data(file_path=data_path)
    print(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")

    print("\nSTEP 2: Fitting and saving preprocessing pipeline...")
    preprocessor = fit_and_save_preprocessor(X_train)

    print("\nSTEP 3: Transforming features for training and testing...")
    X_train_trans = transform_data(preprocessor, X_train)
    X_test_trans = transform_data(preprocessor, X_test)
    print(f"Transformed feature dimension: {X_train_trans.shape[1]} features.")

    models = get_models()
    trained_models = {}

    print("\nSTEP 4: Training Classification Models...")
    print("-" * 60)

    for name, model in models.items():
        print(f"Training [{name}]...")
        start_time = time.time()
        model.fit(X_train_trans, y_train)
        elapsed = time.time() - start_time
        print(f"-> Completed {name} in {elapsed:.2f} seconds.")

        if save_models:
            model_slug = name.lower().replace(" ", "_")
            model_path = os.path.join(MODELS_DIR, f"{model_slug}.joblib")
            joblib.dump(model, model_path)
            print(f"   Saved model to '{model_path}'")

        trained_models[name] = model

    print("-" * 60)
    print("All models successfully trained!\n")
    return trained_models, preprocessor, X_test_trans, y_test, X_test


if __name__ == "__main__":
    train_all_models()
