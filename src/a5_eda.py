"""Part A task 7 -- pandas cross-check of Q2 and exploratory statistics."""
import pandas as pd

o = pd.read_csv("data/clean/orders.csv")
c = pd.read_csv("data/clean/customers.csv")
p = pd.read_csv("data/clean/products.csv")
comp = o[o["status"] == "completed"].merge(c[["customer_id", "city"]], on="customer_id") \
                                    .merge(p[["product_id", "category"]], on="product_id")

# (a) pivot-table rebuild of Q2
pv = pd.pivot_table(comp, values="order_amount", index="city", columns="category",
                    aggfunc="sum", margins=True, margins_name="Total")
print(pv.round(2).to_string())
print(f"\npivot grand total INR {pv.loc['Total', 'Total']:,.2f} (SQL Q2 total INR 2,998,668.80)")

by_city = comp.groupby("city").agg(sum_inr=("order_amount", "sum"),
                                   mean_inr=("order_amount", "mean"),
                                   count=("order_amount", "count")).round(2)
print("\n", by_city.to_string())

# (b) distribution of completed order_amount
a = comp["order_amount"]
q1, q3 = a.quantile(.25), a.quantile(.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\nmean {a.mean():.2f} | median {a.median():.2f} | skew {a.skew():.2f}")
print(f"Q1 {q1:.2f} | Q3 {q3:.2f} | IQR {iqr:.2f} | lower fence {lo:.2f} | upper fence {hi:.2f}")
print(f"outliers below {int((a < lo).sum())} | above {int((a > hi).sum())}")

corr = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print("\ncorrelation matrix (completed orders):\n", corr.round(3).to_string())
pairs = corr.where(~pd.DataFrame(
    [[i <= j for j in range(4)] for i in range(4)], index=corr.index, columns=corr.columns)).stack()
top = pairs.abs().idxmax()
print(f"largest |r|: {top[0]} vs {top[1]} = {pairs[top]:.3f}")
