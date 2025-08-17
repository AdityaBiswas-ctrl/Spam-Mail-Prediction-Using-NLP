from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression with GridSearchCV
    log_reg = LogisticRegression(max_iter=2000)
    param_grid = {'C': [0.01, 0.1, 1.0, 10], 'penalty': ['l2']}
    grid_lr = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
    grid_lr.fit(X_train, y_train)
    y_pred_lr = grid_lr.predict(X_test)
    print("=== Logistic Regression ===")
    print("Best Params:", grid_lr.best_params_)
    print(classification_report(y_test, y_pred_lr))
    print("Accuracy:", accuracy_score(y_test, y_pred_lr))

   
