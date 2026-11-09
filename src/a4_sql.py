"""Part A task 6 -- run the seven queries in sql/queries.sql against data/instamart.db."""
import re
import sqlite3

import pandas as pd


def load_queries(path="sql/queries.sql"):
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"^-- (Q\d):.*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip().rstrip(";") for i in range(1, len(parts), 2)}


def main():
    q = load_queries()
    con = sqlite3.connect("data/instamart.db")
    r = {k: pd.read_sql_query(v, con) for k, v in q.items()}
    con.close()

    print(f"Q1 customers with >INR 8000 in completed orders above INR 500: {len(r['Q1'])}")

    q2 = r["Q2"]
    print(f"Q2 rows: {len(q2)}; total revenue INR {q2['revenue'].sum():,.2f}; "
          f"largest: {q2.iloc[0]['city']} / {q2.iloc[0]['category']} INR {q2.iloc[0]['revenue']:,.2f}")

    print(f"Q3 products above average price: {len(r['Q3'])}")
    print(f"Q4 customers with no completed order: {len(r['Q4'])}")
    print(f"Q5 customers above average spend: {len(r['Q5'])}; average INR {r['Q5']['avg_total'].iloc[0]:,.2f}")

    q6 = r["Q6"]
    print("Q6 top customer per city:")
    print(q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
    diff = q6[q6["rnk"] != q6["row_num"]]
    print(f"Q6 rows where RANK() != ROW_NUMBER(): {len(diff)}")
    print(diff.to_string(index=False))

    q7 = r["Q7"]
    print(f"Q7 daily rows: {len(q7)}; last cumulative INR {q7['cumulative_revenue'].iloc[-1]:,.2f}")


if __name__ == "__main__":
    main()
