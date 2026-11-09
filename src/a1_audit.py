"""Part A task 2 -- audit the raw exports (shape, dtypes, missing values, planted problems)."""
import pandas as pd

RAW = "data/raw"


def missing_report(df, name):
    print(f"\n=== {name}: shape {df.shape} ===")
    rep = pd.DataFrame({"dtype": df.dtypes.astype(str),
                        "missing": df.isna().sum(),
                        "missing_pct": (df.isna().mean() * 100).round(2)})
    print(rep.to_string())


def main():
    orders = pd.read_csv(f"{RAW}/orders.csv", keep_default_na=False, na_values=[""])
    customers = pd.read_csv(f"{RAW}/customers.csv")
    products = pd.read_csv(f"{RAW}/products.csv")
    tickets = pd.read_csv(f"{RAW}/tickets.csv", keep_default_na=False)

    for df, name in [(customers, "customers.csv"), (products, "products.csv"),
                     (orders, "orders.csv"), (tickets, "tickets.csv")]:
        missing_report(df, name)

    amt_num = pd.to_numeric(orders["order_amount"], errors="coerce")
    deliv = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    age = pd.to_numeric(customers["age"], errors="coerce")
    gender = customers["gender"].dropna()

    audit = [
        ("raw orders rows", len(orders)),
        ("exact duplicate rows", int(orders.duplicated().sum())),
        ("rows with status pending", int((orders["status"] == "pending").sum())),
        ("distinct order_id", orders["order_id"].nunique()),
        ("non-numeric order_amount", int((amt_num.isna() & orders["order_amount"].notna()).sum())),
        ("negative delivery_minutes", int((deliv < 0).sum())),
        ("blank delivery_minutes", int(orders["delivery_minutes"].isna().sum())),
        ("blank age", int(customers["age"].isna().sum())),
        ("age outside 10-100", int(((age < 10) | (age > 100)).sum())),
        ("distinct raw city spellings", customers["city"].nunique()),
        ("distinct non-blank raw gender spellings", gender.nunique()),
        ("blank gender", int(customers["gender"].isna().sum())),
    ]
    print("\n=== Audit figures ===")
    print(pd.DataFrame(audit, columns=["check", "value"]).to_string(index=False))


if __name__ == "__main__":
    main()
