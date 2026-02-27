from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import optuna
import job6 # Wait, it's joblib
import joblib
import os

def train_model(pipeline, X_train, y_train):
    """Fit the model pipeline on training data."""
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    """Evaluate the model and return a dictionary of metrics."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline[-1], 'predict_proba') else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    
    return metrics

def run_cross_validation(pipeline, X, y, cv_folds=5):
    """Perform Stratified K-Fold Cross Validation."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy')
    return scores

def tune_hyperparameters(objective_func, n_trials=50):
    """Run Optuna study to find best hyperparameters."""
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_func, n_trials=n_trials)
    return study.best_params, study.best_value

def save_model(pipeline, filename, config):
    """Save the model pipeline to the models directory."""
    models_path = config.get('paths', {}).get('models', 'models/')
    os.makedirs(models_path, exist_ok=True)
    path = os.path.join(models_path, filename)
    joblib.dump(pipeline, path)
    return path
