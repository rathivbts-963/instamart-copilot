# Instamart Customer Intelligence & Support Copilot

## Overview

An internal Copilot for the (fictional) Swiggy Instamart Head of Customer Experience, built in four connected Parts. All data is synthetic and was produced by `generate_data.py`. All money is in Indian Rupees (INR). Every language-model call goes through the offline `mock_llm.py`, so no API key, network or paid service is used anywhere.

| Part | What it does | Main outputs |
|---|---|---|
| A | Audits and cleans the raw exports, stores them in SQLite, and answers business questions in SQL and pandas | `data/clean/*.csv`, `data/instamart.db`, `sql/queries.sql` |
| B | Builds RFM customer segments (K-means) and a leakage-free lapse-prediction model | `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv` |
| C | Triages support tickets with guardrails, PII redaction, retries, output validation, a token budget and segment-aware routing | `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv` |
| D | Runs a TF-IDF policy RAG assistant and a ReAct refund agent with tools, safety limits and checkpoint/resume | `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/` |

Each Part reads what the earlier Parts produced. Part B reads Part A's clean CSVs. Part C routes tickets using Part B's segments. Part D's agent queries Part A's database and reuses Part D's own retrieval.

Repository layout:

```
generate_data.py, mock_llm.py     supplied files (unmodified)
run_all.sh                        runs every script below in order
sql/queries.sql                   seven SQL queries (Part A)
src/a1_audit.py                   raw-data audit
src/a2_clean.py                   cleaning + CSV/SQLite export
src/a3_validate.py                checkpoint assertions -> "ALL CHECKPOINTS PASSED"
src/a4_sql.py                     runs sql/queries.sql with pd.read_sql_query
src/a5_eda.py                     pandas cross-check and exploratory statistics
src/b1_segments.py                RFM, K-means, PCA, segment file
src/b2_lapse.py                   lapse label, features, tuning, balancing, model, retention list
src/c_triage.py                   ticket triage pipeline
src/d_rag.py                      chunking, TF-IDF retrieval, answer(), evaluation
src/d_agent.py                    tools, ReAct loop, limits, checkpoint/resume, demos
```

## Reproduce

These commands need Python 3.11 or newer. Run them from the repository root after cloning. On Windows, step 2 is `.venv\Scripts\activate`.

1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python generate_data.py`  → last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`
5. `python src/a1_audit.py`
6. `python src/a2_clean.py`
7. `python src/a3_validate.py`  → `ALL CHECKPOINTS PASSED`
8. `python src/a4_sql.py`
9. `python src/a5_eda.py`
10. `python src/b1_segments.py`
11. `python src/b2_lapse.py`
12. `python src/c_triage.py`
13. `python src/d_rag.py`
14. `python src/d_agent.py`  → `ALL AGENT CHECKS PASSED`

`./run_all.sh` runs steps 4–14 in one go. Every random step uses a fixed seed (`random_state=42`, and `random.Random(42)` for the retry jitter).

## Part A

### Audit of the raw exports

| Table | Shape | Columns with missing values (count, %) |
|---|---|---|
| customers.csv | 1,200 × 5 | age 61 (5.08%), gender 30 (2.50%); others 0 |
| products.csv | 40 × 4 | none |
| orders.csv | 13,084 × 9 | delivery_minutes 2,541 (19.42%); others 0 (order_amount holds the text values `N/A` / `error`, read as text) |
| tickets.csv | 164 × 5 | none when read with `keep_default_na=False` (4 texts are empty strings, 8 order_ids are empty strings) |

Data types: in customers, `age` is a float (because of the blanks) and the other columns are text. In products, `price_inr` is an integer. In orders, `quantity` and `discount_pct` are integers, `delivery_minutes` is a float, and `order_amount` and `order_ts` are text. In tickets, every column is text.

Orders are read with `keep_default_na=False, na_values=[""]`. Without this, pandas silently turns the literal `N/A` into a missing value, and only 67 of the 120 bad amounts are detected.

| Audit figure | Value |
|---|---|
| raw orders.csv rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status `pending` | 656 |
| distinct order_id | 12,309 |
| order_amount not convertible to a number | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| blank age | 61 |
| age outside 10–100 | 6 |
| distinct raw city spellings | 24 |
| distinct non-blank raw gender spellings | 10 |
| blank gender | 30 |

### Cleaning rules (applied in this order) and what each changed

| # | Rule | Rows / values changed |
|---|---|---|
| C1 | city: strip spaces and apply title case, leaving exactly 6 cities | 476 values |
| C2 | gender: lower-case and map `m`/`male` → male, `f`/`female` → female, blank → unknown | 950 values (920 respellings + 30 blanks → unknown) |
| C3 | age outside 10–100 set to missing | 6 values |
| C4 | missing age filled with the median age of the customer's city (`groupby('city').transform('median')`) | 67 values (61 blank + 6 impossible) |
| C5 | signup_date converted to datetime | 1,200 |
| O1 | order_ts converted to datetime | 13,084 |
| O2 | exact duplicate rows removed | 119 rows |
| O3 | for each order_id keep only the record with the latest order_ts (drops the earlier `pending` copies) | 656 rows |
| O4 | order_amount converted to numeric (unparseable → missing), then filled with `round(price_inr × quantity × (1 − discount_pct/100), 2)` | 112 values (the other 8 bad amounts sat in rows already removed by O2/O3) |
| O5 | negative delivery_minutes set to missing | 105 values (1 of the 106 was in a removed duplicate) |
| O6 | completed orders: missing delivery_minutes filled with the median of that product category among completed orders (`groupby('category').transform('median')`). Cancelled orders stay missing, because a cancelled order was never delivered and imputing a time would invent data. | 774 values filled; 1,199 cancelled left missing |

Results after cleaning:
- **Customers:** Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Genders are female 610, male 560, unknown 30. No ages are missing, and every age is between 18 and 60.
- **Orders:** 12,309 rows, one per order_id: 11,110 completed and 1,199 cancelled, with no pending rows. No order_amount is missing. Completed orders have no missing delivery_minutes; all 1,199 cancelled orders do. Completed revenue totals INR 2,998,668.80.

### Store and validate

The clean tables are written to `data/clean/{customers,products,orders}.csv` and to `data/instamart.db` as tables `customers`, `products` and `orders`. `order_ts` is stored as text `YYYY-MM-DD HH:MM:SS`. `src/a3_validate.py` asserts 13 checkpoints and prints **ALL CHECKPOINTS PASSED**: city counts, gender counts, ages, row count, uniqueness, status counts, no pending rows, no missing amounts, delivery gaps, revenue total, database tables, database row count and timestamp format.

### SQL analysis (`sql/queries.sql`, run by `src/a4_sql.py`)

| Query | Technique | Result |
|---|---|---|
| Q1 | WHERE then HAVING | 30 customers have more than INR 8,000 in completed orders that are each above INR 500 |
| Q2 | three-table join | 30 city × category rows summing to INR 2,998,668.80; largest is Bengaluru / Electronics at INR 233,909.45 |
| Q3 | scalar subquery | 11 products priced above the average catalogue price |
| Q4 | LEFT JOIN | 25 customers have no completed order |
| Q5 | two CTEs | 550 customers spend more than the average of INR 2,552.06 |
| Q6 | RANK / DENSE_RANK / ROW_NUMBER | see below; 8 rows have RANK() ≠ ROW_NUMBER() |
| Q7 | running SUM() OVER | 183 daily rows; last cumulative value INR 2,998,668.80 |

Q6 top customer per city:

| City | Customer | Completed spend (INR) |
|---|---|---|
| Bengaluru | C0082 | 12,856.20 |
| Chennai | C0930 | 10,151.40 |
| Delhi | C0291 | 13,742.60 |
| Hyderabad | C0743 | 12,730.05 |
| Mumbai | C0693 | 14,766.55 |
| Pune | C0426 | 10,377.10 |

When two customers in the same city have exactly the same total spend, RANK() gives them the same number while ROW_NUMBER() always counts on (1, 2, 3, …). So the second customer in each tie, and in this data the 8 such rows, gets a RANK() lower than its ROW_NUMBER(). An example is Pune C0057 and C0936, both at INR 285.00: both are rank 189, with row numbers 190 and 191.

### pandas cross-check and exploratory analysis

`pd.pivot_table(..., margins=True)` rebuilds Q2. Its grand total is **INR 2,998,668.80**, identical to the SQL total.

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed order_amount statistics: mean INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80, Q3 INR 297.00, IQR INR 232.20. The 1.5×IQR fences are INR −283.50 and INR 645.30, giving 0 outliers below and 897 above.

Correlations over completed orders: order_amount–discount_pct −0.095, order_amount–quantity +0.087, order_amount–delivery_minutes −0.003, discount_pct–delivery_minutes −0.010, quantity–delivery_minutes −0.013, discount_pct–quantity −0.000.

**What is this data about?** It covers six months (Apr–Sep 2025) of synthetic quick-commerce orders: 12,309 orders from 1,200 customers in six cities, across 40 products in five categories. Of these, 11,110 orders were completed, worth INR 2,998,668.80.
**What do the distributions look like?** Order value is strongly right-skewed: the mean (INR 269.91) is about double the median (INR 135.00), and skewness is 4.61. Most baskets are cheap staples. Completed deliveries have a median of 25 minutes, and 8.8% take more than 45 minutes.
**Outliers or anomalies?** 897 orders lie above the INR 645.30 fence and none below. 635 of them are Electronics, so they are genuine high-value purchases and are kept. The real anomalies were negative times, unparseable amounts and duplicate or pending rows, and cleaning fixed them.
**Relationships?** Every correlation is weak. The largest in absolute value is order_amount with discount_pct (−0.095), just ahead of order_amount with quantity (+0.087). A correlation this small explains under 1% of the variance, and correlation of any size does not establish causation.

## Part B

### RFM table (snapshot 2025-10-01 00:00:00, completed orders only)

1,175 customers. Recency: median 19 days, min 0, max 180. Frequency: mean 9.46, max 33. Monetary: mean INR 2,552.06, max INR 14,766.55.

### K-means model selection (StandardScaler on R, F, M; `KMeans(n_init=10, random_state=42)`)

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Chosen K = 4.** The elbow is clearest at K=4: going from 3 to 4 clusters cuts inertia by 41% (1268.2 → 743.1), while going from 4 to 5 cuts only 17% (743.1 → 617.4). K=4 also has the best silhouette (0.526), and four groups map cleanly onto four distinct campaigns.

Inertia for K=8 is always lower than for K=2, because adding centroids can only bring each point closer to its nearest centre. Inertia keeps falling until it reaches 0 when every point is its own cluster. Minimising inertia alone would therefore always pick the largest K, so we judge by where the gains flatten (the elbow), by cluster separation (silhouette) and by whether the business can act on the groups.

### Segment profiles (K = 4)

The cluster numbers are only labels. They carry no ordering, and a different seed could renumber them.

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 1 | **Premium Big-Spenders** (is_premium = True) | 138 | 24.12 | 6.77 | 7,129.70 | VIP care: priority support, early access to electronics launches |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Loyalty points / subscription on daily staples to keep the habit |
| 0 | Steady Occasionals | 436 | 28.86 | 3.77 | 1,057.62 | Basket-building nudges (free-delivery threshold, bundles) |
| 3 | Drifting / At-Risk | 182 | 109.04 | 2.20 | 685.99 | Time-limited win-back coupon and a "we miss you" push |

PCA(2) on the scaled RFM matrix: `explained_variance_ratio_` = 0.634 and 0.218, a total of 0.852. Two components keep 85% of the RFM variance.
`outputs/customer_segments.csv` has the columns customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium. is_premium is True only for the highest-mean-monetary segment.

### Lapse problem

- **Cutoff:** 2025-09-01. The cohort is the 1,156 customers with a completed order before the cutoff. `lapsed = 1` if the customer has no completed order from 2025-09-01 to 2025-09-30. 443 customers lapsed (38.32%).
- **Features** (computed only from orders dated before the cutoff, plus the customer table):
  - recency, frequency, monetary
  - average discount_pct
  - cancellation rate (cancelled orders ÷ all final-status orders)
  - average delivery_minutes of completed orders
  - favourite category (most frequent category; ties broken alphabetically)
  - city, age, gender
  - tenure_days (cutoff − signup_date)
- **No leakage:** no order dated on or after 2025-09-01 enters any feature, and the segment labels from task 3 are not used as features.
- **Split:** `train_test_split(test_size=0.2, stratify=y, random_state=42)` gives 924 training and 232 test rows.
- **Baseline** (always predict "not lapsed") on the test set: accuracy 61.6%, lapsed-class precision, recall and F1 all 0.

### Preprocessing without leakage

A `ColumnTransformer` runs inside a scikit-learn `Pipeline`. It applies `StandardScaler` to the 8 numeric features and `OneHotEncoder(drop="first")` (k−1 columns) to city, gender and favourite category. **No encoder, scaler or resampler ever saw a test row.** The pipelines (and GridSearchCV's inner folds) are fitted on training rows only, and oversampling is applied only to the training rows.

The fitted scaler's mean of frequency is **8.1147**, while the full-cohort mean is **8.0285**. They differ because the scaler learned only from the 924 training customers, not from the 232 test customers. The code asserts that no feature column contains a missing value.

### Tuning and test evaluation (lapsed class = positive)

| Model | Grid | Best params | Best CV F1 |
|---|---|---|---|
| LogisticRegression(max_iter=1000) | C ∈ {0.01, 0.1, 1, 10, 100} | C = 100 | 0.651 |
| RandomForest(n_estimators=100, random_state=42) | max_depth ∈ {3,5,8} × min_samples_leaf ∈ {5,10,20} (45 fits) | max_depth 8, min_samples_leaf 10 | 0.681 |

Test set, evaluated once after tuning finished:

| Model | Confusion matrix [[TN, FP], [FN, TP]] | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| Baseline | [[143, 0], [89, 0]] | 0.000 | 0.000 | 0.000 | 0.616 |
| Logistic regression (C=100) | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 | 0.728 |
| **Random forest (depth 8, leaf 10) — selected** | [[113, 30], [31, 58]] | 0.659 | 0.652 | **0.655** | 0.737 |

The best C differs from the brief's reference (C=1), as the brief allows. Our feature set's scaling leaves C=100 marginally ahead in cross-validation, and test F1 is of the expected order.

**Choice: random forest.** Missing a lapsing customer is the more expensive error, because their future spend (about INR 2,552 per half-year for an average customer) is lost. Wrongly targeting a loyal customer only costs a coupon or a push message. Both models catch the same number of lapsers (recall 0.652, 58 of 89). The random forest does so with 2 fewer wasted offers (30 vs 32 false positives), giving the higher precision and F1 (0.655 vs 0.648). It also needs no linearity assumption and gives readable feature importances. If the retention team wants more recall, the next section shows how to buy it.

### Class-balance experiment (training rows only)

Lapsed training rows were oversampled with replacement (`random_state=42`) until the classes were 570 / 570, and the logistic regression (C=100) was refit. The test set is the original, unchanged one.

| Logistic regression | Precision | Recall | F1 |
|---|---|---|---|
| Unbalanced training | 0.644 | 0.652 | 0.648 |
| Balanced training | 0.589 | **0.854** | 0.697 |

**Recall moved most** (+0.20): the balanced model now catches 76 of 89 lapsers instead of 58. Precision fell slightly, to 0.589, and did not rise. For the retention team, this means about 85% of lapsing customers get reached, at the price of about 4 in 10 offers going to customers who would have stayed. That is a good trade when an offer is cheap compared with a lost customer.

### Interpretation

Random forest top-5 feature importances:

| Feature | Importance | Business reading |
|---|---|---|
| frequency | 0.280 | How often someone orders is the strongest signal of habit |
| monetary | 0.271 | Spend level separates big-basket customers from occasional ones |
| recency | 0.115 | A long gap since the last order is an early warning |
| avg_delivery | 0.061 | Slow deliveries in the past push customers away somewhat |
| cancel_rate | 0.052 | Customers who cancel more behave differently from steady buyers |

Logistic regression coefficient signs (standardised features; categories are compared with the dropped first level):

| Feature | Sign (coef) | Business reading |
|---|---|---|
| frequency | negative (−1.698) | Each additional order before the cutoff sharply lowers the lapse risk |
| fav_category = Electronics | negative (−1.137) | Electronics-led (premium-type) customers are less likely to lapse than dairy-led ones with the same history |
| monetary | positive (+0.285) | At equal frequency, higher spend means a few large baskets rather than a habit, so the risk is slightly higher |

**Persistence:** the selected pipeline is saved with `joblib.dump` to `outputs/lapse_model.joblib`. When reloaded with `joblib.load`, it reproduces the original 232 test predictions exactly (asserted in code).

**Retention list:** the test set contains 40 Premium-segment customers. The top 10 by predicted lapse probability (0.450–0.491) are in `outputs/retention_list.csv`, and **2 of these 10 truly lapsed** (C0525, C0565). The Premium flag comes from whole-window RFM that includes the September label window. It is a post-hoc label, so this retention list is illustrative and not a leakage-free evaluation.

## Part C

### Guardrails and redaction

- 4 empty tickets get `skipped_empty`, and 4 long tickets (734 estimated tokens each, above MAX_INPUT_TOKENS = 400) get `flagged_long`. None of them is ever sent to the model.
- Redaction is applied in this order: email → `[EMAIL]`, phone → `[PHONE]`, order id → `[ORDER_ID]`.

| Type | Substitutions |
|---|---|
| Email | 55 |
| Phone | 91 |
| Order id | 132 |
| **Total** | **278** |

Re-applying the three patterns to the redacted texts finds **0** matches. Before every model call, an assertion checks that no email, phone or order id is present in the prompt. The user message is exactly the redacted ticket text.

### Prompts

The zero-shot prompt (43 estimated tokens) contains CLASSIFY, the four categories and a JSON-only reply format, but no `EXAMPLES:`. The few-shot prompt (104 estimated tokens) adds an `EXAMPLES:` line with three labelled examples: fraud, technical and billing.

### Resilient calls

Every call goes through `call_with_retry`: at most 3 retries per ticket, with a wait of `0.05 × 2**n × uniform(0.9, 1.15)` and one log line per retry. `mock_llm.reset()` runs before each full run. A full run over the 156 eligible tickets hits **19 TransientLLMErrors, all recovered on retry 0**. The waits range from 0.0453 to 0.0562 s, inside the allowed 0.045–0.0575 s, and no ticket ends with a permanent error.

### Output validation (one full run, 156 eligible tickets)

| Status | Count |
|---|---|
| classified_json | 135 |
| classified_keyword (chatty reply, exactly one category word via `re.findall`) | 12 |
| flagged_invalid_output (e.g. JSON with category `shipping`) → human review | 9 |

### Zero-shot vs few-shot (no budget)

| Run | Classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126/147) | 11,116 | 75.6 |
| Few-shot | 147 | 100.0% (147/147) | 20,632 | 140.4 |

Zero-shot confusion table (rows = true category, columns = predicted category):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

All 21 zero-shot errors are fraud tickets (11) or technical tickets (10) predicted as billing. The few-shot prompt costs about 65 more tokens per classified ticket (140.4 vs 75.6, +86%), or 9,516 extra tokens for the whole run. In return it fixes all 21 errors. Eleven of those were fraud reports that zero-shot would have sent to the billing queue instead of the security team, which has a 2-hour deadline. Whenever a misroute is costly (fraud, security, premium SLAs), these extra tokens are clearly worth it. Zero-shot only makes sense for low-stakes, high-volume batches where a human re-checks the output anyway.

### Budget and routing

- **Budget demo** (`outputs/triage_budget_demo.csv`): zero-shot run with TOKEN_BUDGET = 6000. The run spent 6,010 tokens, after which **72 tickets ended `budget_exhausted`** and went to human review.
- **Final triage** (`outputs/triage_results.csv`, 164 rows): few-shot run with TOKEN_BUDGET = 100000, which spent 20,632 tokens. The routing rules come from `outputs/customer_segments.csv`; a customer missing from the file counts as not premium.

| Route | Priority | Tickets |
|---|---|---|
| human_review (4 empty + 4 long + 9 invalid output) | NORMAL | 17 |
| security_team (fraud) | HIGH | 34 |
| premium_desk (is_premium) | HIGH | 11 |
| support_queue | NORMAL | 102 |

**Which outputs were rule-validated and which went to humans:**
- **Rule-validated (147 tickets):** 135 replies were parsed as JSON with an allowed category, and 12 were accepted by the exact-one-category-word rule. Only these were routed automatically.
- **Sent to humans (17 tickets):** the 8 tickets blocked by input guardrails and the 9 replies that failed validation. No category was ever guessed for them.

## Part D

### Chunking and retrieval

Documents are split into 40-word chunks with a 10-word overlap (step 30), stopping after the first chunk that reaches the last word. This gives **15 chunks**: refund 4, delivery 3, escalation 3, payment 3, privacy 2. `TfidfVectorizer()` is fitted on the chunk texts, and `retrieve(query, k=3)` ranks chunks by cosine similarity. For the query "refund policy late delivery", the top three chunks are all from refund_policy.txt (scores 0.272, 0.251, 0.245).

TF (term frequency) measures how often a word occurs in a chunk. IDF (inverse document frequency) gives more weight to words that appear in few chunks, so rare, specific words like "wallet" count more than common ones like "order". One limitation: TF-IDF only matches exact words. A question like "Can I get my money back for a slow order?" shares almost no words with "full refund … delivered in more than 45 minutes", even though they mean the same thing.

### Grounded answering

`answer(question)` retrieves 3 chunks and sends a system message containing **POLICY ASSISTANT**. It tells the model to answer only from the context and otherwise to say it could not find the information. The user message has the form `CONTEXT:\n[<file>]\n<chunk>\n…\nQUESTION:\n<question>`. The function returns the answer text and the sources.

### Evaluation (`data/raw/policy_eval.csv`)

| Metric | Overlap 10 (15 chunks) | Overlap 0 (13 chunks) |
|---|---|---|
| expected_doc in top-3 sources (12 answerable) | 12/12 | 12/12 |
| answer contains expected_fact (12 answerable) | 12/12 | **10/12** |
| fallback "I could not find…" (2 unanswerable) | 2/2 | 2/2 |

Two questions break with overlap 0:
- **"What delivery time makes a completed order eligible for a full refund?"** The sentence "If a completed order is delivered in more than 45 minutes, the customer receives a full refund…" is cut between word 39 ("customer") and word 40 ("receives"). One chunk holds "more than 45 minutes" and the next holds "receives a full refund of the order amount.". The assistant returns the second fragment, which lacks the expected fact.
- **"How soon is money returned when a payment fails but is deducted?"** "…returned automatically | within 48 hours" is split the same way.

The 10-word overlap keeps both sentences whole in at least one chunk.

### Tools

`TOOLS_SPEC` lists five tools, each with a name, a one-sentence description and a JSON-style parameter schema. `TOOL_MAP` maps each name to its Python function.

| Tool | Behaviour |
|---|---|
| `get_order(order_id)` | Reads `data/instamart.db`; returns order_id, status, order_amount, delivery_minutes, or `{"error": "order not found"}` |
| `search_policy(query)` | Returns `{"results": retrieve(query)}`, the same retriever as the policy assistant |
| `calculate_refund(order_amount, delivery_minutes)` | Delivery above 45 min → 100% of order_amount; otherwise (or no delivery) → 0 |
| `issue_refund(order_id, refund_inr)` | Appends a timestamped line to `outputs/refund_ledger.jsonl` |
| `escalate_to_human(order_id, refund_inr)` | Appends a timestamped line to `outputs/refund_ledger.jsonl` (refunds above INR 1000) |

### ReAct loop: seven scenarios (defaults: max_iterations=6, token_limit=6000, time_limit_seconds=10)

| Scenario | Expected | Status | Refund (INR) | Model calls | Tokens used | Tools executed |
|---|---|---|---|---|---|---|
| S1 | REFUND | REFUND | 128.25 | 5 | 1,142 | get_order → search_policy → calculate_refund → issue_refund |
| S2 | REFUND | REFUND | 135.00 | 5 | 1,142 | get_order → search_policy → calculate_refund → issue_refund |
| S3 | ESCALATE | ESCALATE | 1,999.00 | 5 | 1,129 | get_order → search_policy → calculate_refund → escalate_to_human |
| S4 | DENIED | DENIED | 0.00 | 4 | 801 | get_order → search_policy → calculate_refund |
| S5 | DENIED | DENIED | 0.00 | 3 | 484 | get_order → search_policy |
| S6 | NOT_FOUND | NOT_FOUND | 0.00 | 2 | 170 | get_order |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,294 | search_policy × 6 |

Every run is appended to `outputs/agent_runs.jsonl` with its scenario_id, status, final_answer, llm_calls, tokens_used and every step's tool, arguments and observation. The three demonstration runs below are logged there too, with a `note` field.

`outputs/refund_ledger.jsonl` records four actions:
- issue_refund ORD-100048 for INR 128.25 (S1)
- issue_refund ORD-100067 for INR 135.00 (S2)
- escalate_to_human ORD-100145 for INR 1,999.00 (S3)
- a second issue_refund ORD-100048 for INR 128.25, from the crash-and-resume demonstration

### Stop safely

| Demonstration | Status |
|---|---|
| S7 with defaults (6 model calls, 3,294 tokens, under the 6,000 limit) | **FALLBACK_MAX_ITERATIONS** |
| S1 with token_limit=200 (stops after call 3 at 501 tokens; the requested calculate_refund is **not** executed) | **FALLBACK_TOKEN_LIMIT** |
| S1 with time_limit_seconds=0 (stops before the first model call, 0 calls) | **FALLBACK_TIMEOUT** |

Every fallback returns the steps completed so far, and its final_answer is a human-handoff summary. For example: `HANDOFF to human agent: time limit of 0s reached. Request: '…'. Tools already executed: none.`

### Crash and resume (S1)

After every tool observation, the agent writes a checkpoint `state/<run_id>.json`. It holds the run id, scenario id, limits, the full message list, tokens_used, llm_calls, the step number, the tools executed and the steps. In the demonstration, a simulated crash is raised right after the 2nd tool's checkpoint is saved, and then `resume_agent("S1-crash-resume-demo")` continues from that checkpoint.

Tools executed across the crash and the resume, in order:
1. `get_order` (before crash)
2. `search_policy` (before crash, then crash)
3. `calculate_refund` (after resume)
4. `issue_refund` (after resume)

That is four executions with none repeated. The final status is **REFUND, INR 128.25**.

### ReAct versus plan-and-execute

My ReAct agent makes one model call per step: 5 calls for S1 (1,142 tokens), 3 for S5, and 2 for S6. Because the full message history is re-sent every time, tokens grow roughly quadratically with the number of steps. A plan-and-execute agent would ask the model once for the whole list (get_order → search_policy → calculate_refund → issue_refund), run the tools without further model calls, and call the model again only to replan or summarise. That is about 2 calls for S1 and far fewer tokens.

The cost of planning up front shows on unexpected observations. In S6, get_order returns NOT_FOUND, and in S5 the order is cancelled. A fixed plan would continue into calculate_refund and could even issue a refund unless every step is guarded by a replanning check. ReAct looks at each observation before choosing the next action, so it stops after 2 calls for S6 and denies after 3 for S5 without executing anything wrong.

For this refund workflow, I would keep ReAct. The workflow is short (at most 5 calls, about 1,100 tokens), and it moves money, so reacting to every observation is worth more than the few hundred tokens saved. Its main risk, looping, is already contained by the max-iteration, token and time limits shown above. Plan-and-execute would suit long, predictable batch jobs where observations rarely change the plan.

## Limitations and Next Steps

- **Retention list is illustrative.** The Premium flag is a post-hoc label computed from whole-window RFM, which includes the September window that defines the lapse label. The retention list is therefore illustrative and not a leakage-free evaluation. A production version should compute segments using only pre-cutoff data.
- **Mock model.** All "LLM" behaviour comes from the deterministic `mock_llm.py`, so the accuracy, token and agent figures describe the pipeline's plumbing, not a real model. Swapping in a hosted model would need re-evaluation and a real tokenizer instead of `est_tokens`.
- **Lapse model is modest.** F1 is about 0.65 on a 232-row test set from six months of synthetic data, so it should be retrained on longer history. Tuning the probability threshold (or using the balanced variant) would let the team pick its own recall/precision trade-off.
- **Retrieval is lexical.** TF-IDF misses paraphrases, and fixed 40-word chunks can cut sentences. Sentence-aware chunking or embedding-based retrieval would make it more robust.
- **Refund rules are simplified.** The 7-day request window, payment method and double-refund protection are not enforced; the crash-resume demo deliberately issues S1's refund a second time in the ledger. A real system needs an idempotency key per order.
- **Cleaning choices are imputations.** City-median ages and category-median delivery times smooth the data. An `is_imputed` flag column would let analysts exclude them.
