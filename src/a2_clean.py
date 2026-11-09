"""Part A tasks 3-5 -- clean customers and orders, write data/clean/*.csv and data/instamart.db."""
import os
import sqlite3

import pandas as pd

RAW, CLEAN, DB = "data/raw", "data/clean", "data/instamart.db"
GENDER_MAP = {"male": "male", "m": "male", "female": "female", "f": "female"}


def clean_customers(c):
    log = []
    before = c["city"].copy()
    c["city"] = c["city"].str.strip().str.title()
    log.append(("city: strip + title case", int((before != c["city"]).sum())))

    raw_g = c["gender"].copy()
    c["gender"] = c["gender"].fillna("").str.strip().str.lower().map(GENDER_MAP).fillna("unknown")
    log.append(("gender: mapped to male/female/unknown (values changed)", int((raw_g != c["gender"]).sum())))

    c["age"] = pd.to_numeric(c["age"], errors="coerce")
    bad = (c["age"] < 10) | (c["age"] > 100)
    log.append(("age: outside 10-100 set to missing", int(bad.sum())))
    c.loc[bad, "age"] = float("nan")
    n_missing = int(c["age"].isna().sum())
    c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
    log.append(("age: missing filled with city median", n_missing))

    c["signup_date"] = pd.to_datetime(c["signup_date"])
    log.append(("signup_date: converted to datetime", len(c)))
    return c, log


def clean_orders(o, products):
    log = []
    o["order_ts"] = pd.to_datetime(o["order_ts"])
    log.append(("order_ts: converted to datetime", len(o)))

    n = len(o)
    o = o.drop_duplicates()
    log.append(("exact duplicate rows removed", n - len(o)))

    n = len(o)
    o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
    log.append(("older records of same order_id removed (keep latest order_ts)", n - len(o)))

    o = o.merge(products[["product_id", "category", "price_inr"]], on="product_id", how="left")
    o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
    miss = o["order_amount"].isna()
    o.loc[miss, "order_amount"] = (o.loc[miss, "price_inr"] * o.loc[miss, "quantity"]
                                   * (1 - o.loc[miss, "discount_pct"] / 100)).round(2)
    log.append(("order_amount: unparseable filled from price x qty x (1-discount)", int(miss.sum())))

    o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")
    neg = o["delivery_minutes"] < 0
    o.loc[neg, "delivery_minutes"] = float("nan")
    log.append(("delivery_minutes: negative set to missing", int(neg.sum())))

    comp = o["status"] == "completed"
    cat_median = o[comp].groupby("category")["delivery_minutes"].transform("median")
    fill = comp & o["delivery_minutes"].isna()
    o.loc[fill, "delivery_minutes"] = cat_median[fill]
    log.append(("delivery_minutes: completed orders filled with category median", int(fill.sum())))

    o = o.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)
    return o, log


def main():
    os.makedirs(CLEAN, exist_ok=True)
    customers = pd.read_csv(f"{RAW}/customers.csv")
    products = pd.read_csv(f"{RAW}/products.csv")
    orders = pd.read_csv(f"{RAW}/orders.csv", keep_default_na=False, na_values=[""])

    customers, log_c = clean_customers(customers)
    orders, log_o = clean_orders(orders, products)

    print("=== Cleaning log (rule -> rows/values changed) ===")
    for rule, n in log_c + log_o:
        print(f"{n:>6}  {rule}")

    customers.to_csv(f"{CLEAN}/customers.csv", index=False, date_format="%Y-%m-%d")
    products.to_csv(f"{CLEAN}/products.csv", index=False)
    orders.to_csv(f"{CLEAN}/orders.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")

    db_c = customers.assign(signup_date=customers["signup_date"].dt.strftime("%Y-%m-%d"))
    db_o = orders.assign(order_ts=orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    if os.path.exists(DB):
        os.remove(DB)
    with sqlite3.connect(DB) as con:
        db_c.to_sql("customers", con, index=False)
        products.to_sql("products", con, index=False)
        db_o.to_sql("orders", con, index=False)
    print(f"\nwrote {CLEAN}/*.csv and {DB} "
          f"(customers={len(customers)}, products={len(products)}, orders={len(orders)})")


if __name__ == "__main__":
    main()
