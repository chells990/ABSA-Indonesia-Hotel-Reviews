import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


class ABSASVM:
    def __init__(self, aspects):
        self.aspects = aspects
        self.models = {}
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )

    def train(self, X_train, y_train_dict, X_val, y_val_dict):
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_val_vec = self.vectorizer.transform(X_val)

        for aspect in self.aspects:
            print(f"\nTraining SVM for {aspect}".ljust(50, '-'))
            svm = SVC(
                kernel='rbf',
                class_weight='balanced',
                C=1.0,
                gamma='scale',
                random_state=42
            )
            svm.fit(X_train_vec, y_train_dict[aspect])

            val_preds = svm.predict(X_val_vec)
            val_acc = accuracy_score(y_val_dict[aspect], val_preds)
            print(f"Validation Accuracy: {val_acc:.4f}")
            self.models[aspect] = svm

    def save(self, model_dir):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, model_dir/"vectorizer.joblib")
        for aspect in self.aspects:
            joblib.dump(self.models[aspect], model_dir/f"{aspect}.joblib")

    @classmethod
    def load(cls, model_dir, aspects):
        model_dir = Path(model_dir)
        instance = cls(aspects)
        instance.vectorizer = joblib.load(model_dir/"vectorizer.joblib")
        instance.models = {a: joblib.load(model_dir/f"{a}.joblib") for a in aspects}
        return instance