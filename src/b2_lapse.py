"""Part B tasks 4-8 -- lapse label, leakage-free features, tuned models, balancing, persistence, retention list."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF = pd.Timestamp("2025-09-01")
NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]


def build_dataset():
    o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    c = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
    p = pd.read_csv("data/clean/products.csv")
    o = o.merge(p[["product_id", "category"]], on="product_id")

    before = o[o["order_ts"] < CUTOFF]                      # features: strictly before the cutoff
    sept = o[(o["order_ts"] >= CUTOFF) & (o["order_ts"] < "2025-10-01") & (o["status"] == "completed")]
    comp = before[before["status"] == "completed"]

    f = comp.groupby("customer_id").agg(last_ts=("order_ts", "max"), frequency=("order_id", "nunique"),
                                        monetary=("order_amount", "sum"), avg_discount=("discount_pct", "mean"),
                                        avg_delivery=("delivery_minutes", "mean"))
    f["recency"] = (CUTOFF - f["last_ts"]).dt.days
    f["cancel_rate"] = before.groupby("customer_id")["status"].apply(lambda s: (s == "cancelled").mean())
    fav = comp.groupby(["customer_id", "category"]).size().reset_index(name="n") \
              .sort_values(["customer_id", "n", "category"], ascending=[True, False, True]) \
              .drop_duplicates("customer_id").set_index("customer_id")["category"]
    f["fav_category"] = fav
    f = f.join(c.set_index("customer_id")[["city", "age", "gender", "signup_date"]])
    f["tenure_days"] = (CUTOFF - f["signup_date"]).dt.days
    f["lapsed"] = (~f.index.isin(sept["customer_id"])).astype(int)
    return f[NUM + CAT + ["lapsed"]]


def make_pre():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first"), CAT)])


def metrics(y, pred):
    return dict(precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0), accuracy=accuracy_score(y, pred))


def show(name, y, pred):
    m = metrics(y, pred)
    print(f"\n{name}: confusion matrix [[TN FP] [FN TP]] = {confusion_matrix(y, pred).tolist()}")
    print("  lapsed class -> " + ", ".join(f"{k} {v:.3f}" for k, v in m.items()))
    return m


def main():
    df = build_dataset()
    X, y = df[NUM + CAT], df["lapsed"]
    print(f"cohort {len(df)}; lapsed {y.sum()} ({y.mean():.2%})")
    assert X.isna().sum().sum() == 0, "missing values in features"
    print("no feature column contains a missing value: confirmed")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"train {len(Xtr)} / test {len(Xte)}")

    # (a) baseline
    show("Baseline (always 'not lapsed')", yte, np.zeros(len(yte), dtype=int))

    # leakage check: scaler fitted on training rows only
    pre = make_pre().fit(Xtr)
    sc = pre.named_transformers_["num"]
    print(f"\nscaler mean of frequency (train only): {sc.mean_[NUM.index('frequency')]:.4f} "
          f"vs full-cohort mean: {X['frequency'].mean():.4f}")

    # (b) logistic regression
    lr = GridSearchCV(Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000))]),
                      {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(Xtr, ytr)
    print(f"\nLogReg best {lr.best_params_}, CV F1 {lr.best_score_:.3f}")
    # (c) random forest
    rf = GridSearchCV(Pipeline([("pre", make_pre()),
                                ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
                      {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                      cv=5, scoring="f1").fit(Xtr, ytr)
    print(f"RandomForest best {rf.best_params_}, CV F1 {rf.best_score_:.3f}")

    # (d) test evaluation (only after tuning is finished)
    m_lr = show("Tuned LogReg on test", yte, lr.predict(Xte))
    m_rf = show("Tuned RandomForest on test", yte, rf.predict(Xte))
    best = rf if m_rf["f1"] >= m_lr["f1"] else lr
    best_name = "RandomForest" if best is rf else "LogisticRegression"
    print(f"\nselected model: {best_name}")

    # class-balance experiment (training rows only)
    tr = Xtr.assign(lapsed=ytr)
    pos, neg = tr[tr["lapsed"] == 1], tr[tr["lapsed"] == 0]
    extra = pos.sample(len(neg) - len(pos), replace=True, random_state=42)
    bal = pd.concat([tr, extra])
    print(f"\nbalanced training counts: {bal['lapsed'].value_counts().to_dict()}")
    lr_bal = Pipeline([("pre", make_pre()),
                       ("clf", LogisticRegression(max_iter=1000, C=lr.best_params_["clf__C"]))]) \
        .fit(bal[NUM + CAT], bal["lapsed"])
    show("LogReg (balanced training) on test", yte, lr_bal.predict(Xte))

    # interpretation
    rf_est = rf.best_estimator_
    names = rf_est.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(rf_est.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
    print("\nRandomForest top-5 feature importances:\n" + imp.head(5).round(4).to_string())
    coef = pd.Series(lr.best_estimator_.named_steps["clf"].coef_[0],
                     index=lr.best_estimator_.named_steps["pre"].get_feature_names_out())
    print("\nLogReg coefficients (standardised numeric features):\n" +
          coef.reindex(coef.abs().sort_values(ascending=False).index).round(3).to_string())

    # persist + reload
    model = best.best_estimator_
    joblib.dump(model, "outputs/lapse_model.joblib")
    reloaded = joblib.load("outputs/lapse_model.joblib")
    same = np.array_equal(model.predict(Xte), reloaded.predict(Xte))
    print(f"\nreloaded model reproduces test predictions exactly: {same}")
    assert same

    # retention list: premium test customers ranked by lapse probability
    seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
    prem_ids = seg.index[seg["is_premium"]]
    te = Xte[Xte.index.isin(prem_ids)]
    ret = pd.DataFrame({"customer_id": te.index, "lapse_probability": reloaded.predict_proba(te)[:, 1].round(4),
                        "actually_lapsed": yte.loc[te.index].values})
    ret = ret.sort_values("lapse_probability", ascending=False).head(10)
    ret.to_csv("outputs/retention_list.csv", index=False)
    print(f"\npremium customers in test set: {len(te)}; retention list size {len(ret)}; "
          f"truly lapsed {int(ret['actually_lapsed'].sum())}")
    print(ret.to_string(index=False))


if __name__ == "__main__":
    main()
