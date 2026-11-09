"""Part A task 5 -- assert the reference checkpoints on the clean tables and the database."""
import sqlite3

import pandas as pd

c = pd.read_csv("data/clean/customers.csv")
o = pd.read_csv("data/clean/orders.csv")
comp, canc = o[o["status"] == "completed"], o[o["status"] == "cancelled"]

assert c["city"].value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197,
                                              "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}
assert c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}
assert c["age"].isna().sum() == 0 and c["age"].between(18, 60).all()
assert len(o) == 12309 and o["order_id"].is_unique
assert len(comp) == 11110 and len(canc) == 1199
assert (o["status"] == "pending").sum() == 0
assert o["order_amount"].isna().sum() == 0
assert comp["delivery_minutes"].isna().sum() == 0
assert canc["delivery_minutes"].isna().sum() == 1199
assert round(comp["order_amount"].sum(), 2) == 2998668.80

with sqlite3.connect("data/instamart.db") as con:
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"customers", "products", "orders"} <= tables
    assert con.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 12309
    ts = con.execute("SELECT order_ts FROM orders LIMIT 1").fetchone()[0]
    assert len(ts) == 19 and ts[10] == " "          # YYYY-MM-DD HH:MM:SS text

print("ALL CHECKPOINTS PASSED")
