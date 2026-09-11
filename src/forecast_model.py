import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def _clean_returns(tickers, period="2y"):
    data = yf.download(tickers, period=period)["Close"]
    missing_pct = data.isna().mean()
    bad_tickers = missing_pct[missing_pct > 0.05].index.tolist()
    if bad_tickers:
        data = data.drop(columns=bad_tickers)
    data = data.ffill().dropna()
    return data.pct_change().dropna(), list(data.columns)

def build_features(portfolio_returns, loss_percentile=10):
    df = pd.DataFrame({"return": portfolio_returns})
    # All features use .shift(1) or later — meaning they only use information
    # available BEFORE the day being predicted. This avoids lookahead bias,
    # a critical mistake in any financial forecasting model.
    df["lag_1"] = df["return"].shift(1)
    df["lag_2"] = df["return"].shift(2)
    df["roll_mean_5"] = df["return"].shift(1).rolling(5).mean()
    df["roll_std_5"] = df["return"].shift(1).rolling(5).std()
    df["roll_std_10"] = df["return"].shift(1).rolling(10).std()
    df["momentum_5"] = df["return"].shift(1).rolling(5).sum()

    threshold = np.percentile(df["return"].dropna(), loss_percentile)
    df["target"] = (df["return"] < threshold).astype(int)

    df = df.dropna()
    feature_cols = ["lag_1", "lag_2", "roll_mean_5", "roll_std_5", "roll_std_10", "momentum_5"]
    return df, feature_cols, threshold

def train_risk_event_models(tickers, weights, loss_percentile=10, test_size=0.25):
    returns_df, surviving_tickers = _clean_returns(tickers)
    aligned_weights = weights[:len(surviving_tickers)]
    portfolio_returns = (returns_df * aligned_weights).sum(axis=1)

    df, feature_cols, threshold = build_features(portfolio_returns, loss_percentile)
    X = df[feature_cols]
    y = df["target"]

    # shuffle=False preserves time order — training on the past, testing on the future,
    # never testing on data that came before what it trained on.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    tree_model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
    tree_model.fit(X_train, y_train)
    tree_preds = tree_model.predict(X_test)
    tree_probs = tree_model.predict_proba(X_test)[:, 1]

    logit_model = LogisticRegression(max_iter=1000)
    logit_model.fit(X_train, y_train)
    logit_preds = logit_model.predict(X_test)
    logit_probs = logit_model.predict_proba(X_test)[:, 1]

    can_compute_auc = len(set(y_test)) > 1  # AUC is undefined if the test set has only one class

    return {
        "threshold_return": threshold,
        "loss_percentile": loss_percentile,
        "feature_cols": feature_cols,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "event_rate_test": float(y_test.mean()),
        "tree": {
            "accuracy": accuracy_score(y_test, tree_preds),
            "confusion_matrix": confusion_matrix(y_test, tree_preds).tolist(),
            "feature_importances": dict(zip(feature_cols, tree_model.feature_importances_.tolist())),
            "auc": roc_auc_score(y_test, tree_probs) if can_compute_auc else None,
            "latest_probability": float(tree_model.predict_proba(X.iloc[[-1]])[0, 1]),
        },
        "logistic": {
            "accuracy": accuracy_score(y_test, logit_preds),
            "confusion_matrix": confusion_matrix(y_test, logit_preds).tolist(),
            "coefficients": dict(zip(feature_cols, logit_model.coef_[0].tolist())),
            "auc": roc_auc_score(y_test, logit_probs) if can_compute_auc else None,
            "latest_probability": float(logit_model.predict_proba(X.iloc[[-1]])[0, 1]),
        },
        "test_dates": df.index[-len(X_test):].strftime("%Y-%m-%d").tolist(),
        "test_actual": y_test.tolist(),
        "test_tree_probs": tree_probs.tolist(),
    }   