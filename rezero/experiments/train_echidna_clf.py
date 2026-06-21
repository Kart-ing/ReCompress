"""Train the lightweight Echidna classifier (sklearn, local, no GPU).

Learns checkpoint-vs-pass from the cheap features in echidna_features.py, mimicking the LLM
Echidna's decisions collected by gen_echidna_data.py. Saves a joblib bundle (scaler + model +
feature names + threshold) that the inference path loads. Reports held-out agreement with the
LLM so we know how faithful the cheap replacement is.

Run from repo root:  python -m rezero.experiments.train_echidna_clf
"""
from __future__ import annotations
import os, sys, json, argparse

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ACTII = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_REPO, _ACTII):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/echidna/echidna_train.jsonl")
    ap.add_argument("--out", default="data/echidna/echidna_clf.joblib")
    args = ap.parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
    from rezero.echidna_features import FEATURE_NAMES

    rows = [json.loads(l) for l in open(os.path.join(_REPO, args.data))]
    X = np.array([r["features"] for r in rows], dtype=float)
    y = np.array([1 if r["label"] == "checkpoint" else 0 for r in rows])
    print(f"{len(y)} samples | checkpoint rate {y.mean():.1%} | {X.shape[1]} features")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    # class_weight balanced — checkpoints are the minority but the costly-to-miss class
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(scaler.transform(Xtr), ytr)

    pred = clf.predict(scaler.transform(Xte))
    print("\n=== held-out agreement with LLM Echidna ===")
    print(classification_report(yte, pred, target_names=["pass", "checkpoint"], digits=3))
    print("confusion matrix [rows=true, cols=pred]:")
    print(confusion_matrix(yte, pred))
    agree = (pred == yte).mean()
    print(f"\noverall agreement: {agree:.1%}")

    # feature importances (standardized coefficients)
    print("\nstandardized coefficients:")
    for name, c in sorted(zip(FEATURE_NAMES, clf.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {name:28s} {c:+.3f}")

    bundle = {"scaler": scaler, "model": clf, "features": FEATURE_NAMES,
              "threshold": 0.5, "agreement": float(agree)}
    path = os.path.join(_REPO, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
