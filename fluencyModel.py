import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

def train_and_save_model():
    # Load data from generated numpy files
    X = np.load('feat.npy')
    y = np.load('label.npy').ravel()

    # Fix random seed number
    np.random.seed(7)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100, 200],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    # Create SVM classifier
    svm_clf = SVC(random_state=42)

    # Perform Grid Search
    grid_search = GridSearchCV(svm_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model
    best_svm = grid_search.best_estimator_

    # Make predictions
    y_pred = best_svm.predict(X_test_scaled)

    # Print results
    print("Test set accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model and scaler
    joblib.dump(best_svm, 'best_svm_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    print("Model and scaler saved successfully.")

if __name__ == '__main__':
    train_and_save_model()