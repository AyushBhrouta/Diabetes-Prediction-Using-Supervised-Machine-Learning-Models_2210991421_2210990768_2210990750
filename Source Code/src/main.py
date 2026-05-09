# ==========================================
# Main Execution Script
# ==========================================

import numpy as np
from sklearn.model_selection import train_test_split

from preprocess import load_data, preprocess_data
from train_models import get_models, train_models
from evaluate import (
    evaluate_models,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_accuracy
)


def main():
    # Fix random seed
    np.random.seed(42)

    # Load dataset
    df = load_data("../data/diabetes.csv")

    # Preprocess data
    X, y = preprocess_data(df)

    # Train-test split (80:20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train models
    models = get_models()
    trained_models = train_models(models, X_train, y_train)

    # Evaluate models
    results = evaluate_models(trained_models, X_test, y_test)

    print("\n===== Model Performance =====")
    for model, metrics in results.items():
        print(f"\n{model}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    # Use Random Forest for graphs (as in paper)
    rf_model = trained_models["Random Forest"]

    plot_confusion_matrix(rf_model, X_test, y_test)
    plot_roc_curve(rf_model, X_test, y_test)
    plot_pr_curve(rf_model, X_test, y_test)
    plot_accuracy(results)

    print("\nAll graphs saved in /outputs folder")


if __name__ == "__main__":
    main()