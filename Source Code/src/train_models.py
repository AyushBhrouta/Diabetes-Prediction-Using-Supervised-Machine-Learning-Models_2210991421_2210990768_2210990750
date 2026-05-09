# ==========================================
# Model Training Module
# ==========================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def get_models():
    """
    Initialize models with fixed hyperparameters
    (as per controlled experiment in paper)
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "SVM": SVC(kernel='rbf', C=1, probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    return models


def train_models(models, X_train, y_train):
    """
    Train all models
    """
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models