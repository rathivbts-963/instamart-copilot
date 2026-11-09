"""Part B tasks 1-3 -- RFM table, K-means model selection, segment profiles, PCA, customer_segments.csv."""
import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4


def build_rfm():
    o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    comp = o[o["status"] == "completed"]
    rfm = comp.groupby("customer_id").agg(last_ts=("order_ts", "max"),
                                          frequency=("order_id", "nunique"),
                                          monetary=("order_amount", "sum"))
    rfm["recency"] = (SNAPSHOT - rfm["last_ts"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)
    return rfm[["recency", "frequency", "monetary"]].reset_index()


def name_segments(profile):
    """Business names assigned from each cluster's profile (cluster ids themselves carry no order)."""
    names, left = {}, set(profile.index)
    prem = profile["monetary"].idxmax()
    names[prem] = "Premium Big-Spenders"; left.discard(prem)
    loyal = profile.loc[list(left), "frequency"].idxmax()
    names[loyal] = "Loyal Regulars"; left.discard(loyal)
    risk = profile.loc[list(left), "recency"].idxmax()
    names[risk] = "Drifting / At-Risk"; left.discard(risk)
    for k in left:
        names[k] = "Steady Occasionals"
    return names, prem


ACTIONS = {"Premium Big-Spenders": "VIP care: priority support, early access to electronics launches",
           "Loyal Regulars": "Subscription / loyalty points on daily staples to keep the habit",
           "Drifting / At-Risk": "Win-back coupon with a time limit plus a 'we miss you' push",
           "Steady Occasionals": "Basket-building nudges (free-delivery threshold, bundles)"}


def main():
    os.makedirs("outputs", exist_ok=True)
    rfm = build_rfm()
    print(f"RFM customers: {len(rfm)}")
    print(rfm[["recency", "frequency", "monetary"]].describe().round(2).to_string())

    X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append((k, round(km.inertia_, 1), round(silhouette_score(X, km.labels_), 3)))
    print("\nK  inertia  silhouette")
    for r in rows:
        print(f"{r[0]}  {r[1]:>7}  {r[2]}")

    km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
    rfm["segment_id"] = km.labels_
    profile = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                            frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
    names, prem = name_segments(profile)
    profile["segment_name"] = profile.index.map(names)
    profile["action"] = profile["segment_name"].map(ACTIONS)
    print(f"\nSegment profile (K={FINAL_K}; cluster ids are arbitrary labels):")
    print(profile.to_string())

    pca = PCA(n_components=2).fit(X)
    ev = pca.explained_variance_ratio_
    print(f"\nPCA explained_variance_ratio_: {ev[0]:.3f}, {ev[1]:.3f} (total {ev.sum():.3f})")

    rfm["segment_name"] = rfm["segment_id"].map(names)
    rfm["is_premium"] = rfm["segment_id"] == prem
    rfm.to_csv("outputs/customer_segments.csv", index=False)
    print("wrote outputs/customer_segments.csv")


if __name__ == "__main__":
    main()
