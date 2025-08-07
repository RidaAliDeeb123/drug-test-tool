from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class DrugResponsePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def train_models(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X):
        if not self.is_trained:
            raise Exception("Model not trained yet.")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        if not self.is_trained:
            raise Exception("Model not trained yet.")
        preds = self.model.predict(X_test)
        return accuracy_score(y_test, preds)