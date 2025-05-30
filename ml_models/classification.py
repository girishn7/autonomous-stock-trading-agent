from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_regime_classifier(df):
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['regime']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, 'ml_models/regime_classifier.pkl')
    return clf
