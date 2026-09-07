---
title: "FireDucks for Data Consulting: A Complete Offline Reference"
subtitle: "Execution model, pandas-compatibility gotchas, performance tuning, and analytics workflows for autonomous code generation and RAG retrieval"
author: "PrevAI Health"
date: "September 2026"
---

# How to Use This Document

This document is a reference for an AI coding assistant (running locally, possibly offline) that generates Python data-analysis code using **FireDucks**, a compiler-accelerated, pandas-compatible DataFrame library from NEC (`pip install fireducks`, current stable **1.4.4**, released December 2025). Treat every code block here as verified against FireDucks' official documentation as of September 2026.

This document is written to be retrieved by a RAG system, not only read top to bottom. Every `##` subsection below is written to stand on its own: it restates enough context to be understood if it is the *only* chunk retrieved for a query, rather than assuming the reader just read the previous section. When chunking this file for embedding, split on `#` and `##` headers — each resulting chunk is a complete, self-contained unit of knowledge.

Ground rules to follow when generating FireDucks code:

1. **FireDucks is pandas, syntactically.** The DataFrame/Series API is (almost) identical to pandas. Do not relearn pandas syntax — the entire pandas mental model applies. What you must learn from this document is where FireDucks *differs* from pandas: its lazy execution model, its known incompatibilities, and its performance rules.
2. **Import as** `import fireducks.pandas as pd` (or use the import-hook / Jupyter-magic methods in Section 2). Never `import pandas as pd` if the goal is FireDucks acceleration.
3. **FireDucks is lazy.** Operations build an intermediate-language plan; nothing computes until a materializing call happens (`print()`, `.to_csv()`, `.to_parquet()`, `len()`, control-flow that needs a concrete value, etc.). Don't assume timing, error timing, or in-place mutation behave exactly like eager pandas — see Section 3 and Section 4.
4. **Never pass a user-defined function to `.apply()`, `.map()`, or similar** expecting acceleration — FireDucks' compiler cannot optimize arbitrary Python callables, and this is one of the most common performance mistakes. Prefer vectorized/native DataFrame operations. See Section 9.
5. **Always use bracket notation** (`df["col"]`), never attribute notation (`df.col`), when generating FireDucks code — attribute access can silently disable compiler optimizations and is ambiguous with DataFrame methods/attributes.
6. When a task needs a real pandas object (a library that only accepts `pandas.DataFrame`, e.g. some plotting/statistics libraries), call `.to_pandas()` first. Don't assume `isinstance(df, pandas.DataFrame)` is true for a FireDucks object — it is not.
7. When asked for "a data consulting analysis" or "a report," follow the workflow templates in Section 12: profile the data first, state assumptions, then analyze, then summarize findings in business language, not just code output.


# Table of Contents

1. What FireDucks Is and When to Use It
2. Installation and Setup
3. The Execution Model (Lazy Compilation)
4. Pandas Compatibility and Known Differences
5. Core DataFrame Operations for Data Consulting
6. GroupBy, Aggregation, and Pivot Tables
7. Merging, Joining, and Concatenation
8. Missing Data Handling
9. Performance Tuning and Avoiding Fallback
10. FireDucks' Own API Extensions
11. Statistical Analysis Patterns
12. Data Consulting Workflows
13. Charting and Visualization
14. End-to-End Worked Case Study
15. Common Pitfalls and Anti-Patterns
16. Quick-Reference Cheat Sheet
17. FAQ, Licensing, and Support


# 1. What FireDucks Is and When to Use It

FireDucks is a **compiler-accelerated, pandas-compatible DataFrame library** built by NEC. Its pitch is unusually direct: keep the exact pandas API surface data scientists already know, but instead of executing each pandas call eagerly in Python/Cython the way real pandas does, FireDucks compiles the sequence of DataFrame operations into an intermediate representation, applies compiler-style optimizations to that plan (operation reordering, fusion, dead-code elimination), and executes it on a multi-threaded C++ backend built on Apache Arrow. The claimed benefit is pandas-level (or better) ergonomics with order-of-magnitude speedups on many workloads, with no code rewrite beyond the import line.

**When to reach for FireDucks in a consulting engagement:** the client already has a pandas codebase (notebooks, scripts, an internal pipeline) that has become slow as data volume grew, and rewriting it in polars/Spark/DuckDB is out of scope or too disruptive. Swapping `import pandas as pd` for `import fireducks.pandas as pd` (or using the import hook, see Section 2) is the lowest-effort acceleration path available for an existing pandas codebase.

**When NOT to reach for FireDucks:** if the codebase is being written from scratch, or if it makes heavy use of `.apply()` with custom Python functions, DataFrame/Series subclassing, or exotic pandas internals — FireDucks explicitly does not optimize those paths (they fall back to real pandas under the hood, which costs a conversion round-trip and gives you no speedup, only overhead). In a from-scratch project, polars is usually the safer choice for guaranteed performance, since its API is designed around vectorized expressions from the ground up rather than layering acceleration under a pandas-shaped API. FireDucks' core value is *migration* cost, not API design.

**How it compares:** FireDucks is closest in spirit to Modin (both promise "same pandas API, faster") but takes a compiler/IR approach rather than Modin's distributed-execution-engine approach (Ray/Dask backends). Polars is a different API entirely (expression-based, not pandas-compatible) that trades a migration cost for a more predictable, purpose-built performance model.


# 2. Installation and Setup

## 2.1 Installing FireDucks

```bash
pip install fireducks
```

Supported Python versions are **3.9 through 3.13** (Python 3.8 support was dropped as of FireDucks 1.1.0). Supported platforms are **Linux (x86_64, manylinux wheels)** and **macOS (ARM/Apple Silicon)**. There is currently **no native Windows build** — Windows users run FireDucks under WSL. License is the 3-Clause BSD License.

## 2.2 The three ways to enable FireDucks

**Explicit import (preferred for new code you're writing yourself):**

```python
import fireducks.pandas as pd

df = pd.read_csv("sales.csv")  # this is FireDucks, not real pandas
```

**Import hook from the command line (preferred for accelerating an existing script without editing it):**

```bash
python3 -m fireducks.pandas your_script.py
```

This runs `your_script.py` with every `import pandas as pd` inside it (including inside third-party libraries the script imports) transparently redirected to FireDucks — no source changes required.

**Jupyter/IPython magic (preferred for accelerating an existing notebook):**

```python
%load_ext fireducks.pandas
import pandas as pd     # this cell onward, "pandas" resolves to FireDucks
```

Use the import-hook or magic-extension methods specifically when a script or notebook depends on third-party libraries that internally `import pandas` themselves — those internal imports get accelerated too, which the explicit `import fireducks.pandas as pd` method cannot do for code outside your own file.


# 3. The Execution Model (Lazy Compilation)

FireDucks uses a **lazy execution model**: this is the single most important thing to understand about FireDucks, and the source of nearly every surprising behavior a pandas user will hit. Unlike real pandas, which executes each DataFrame method call immediately, FireDucks defers computation.

## 3.1 How the lazy plan is built and executed

Every time a FireDucks method is called, it does not compute a result — it appends more instructions to an internal intermediate-language (IR) program describing the computation. Nothing actually runs yet. This lets FireDucks' compiler see the *whole* planned computation (or as much of it as has been expressed so far) and optimize it as a batch: reordering operations for efficiency (e.g. `df[df["a"] > 10]["b"]` gets rewritten to select column `"b"` first, the way an expert pandas programmer would hand-tune it), fusing steps together, and dropping dead code.

Materialization (actually running the accumulated IR program) is triggered by specific operations, notably:

- Writing output: `df.to_csv(...)`, `df.to_parquet(...)`
- Displaying/inspecting a result: `print(df)`, showing a DataFrame's repr
- Any call that needs a concrete Python value to proceed (e.g. control flow that branches on a computed value, or `len(df)`)

Until one of these happens, a chain of FireDucks calls like `pd.read_csv(...).sort_values("a").groupby("b").sum()` has done essentially no real computation — it has only built a plan.

## 3.2 Forcing evaluation explicitly

For benchmarking or debugging, you can force materialization mid-chain instead of waiting for a natural trigger:

```python
df = pd.read_csv("data.csv")._evaluate()
df = df.sort_values("a")._evaluate()
```

`_evaluate()` runs the intermediate-language program accumulated so far and returns a materialized result you can safely time in isolation — useful when you want to measure the cost of one specific step rather than the whole lazy chain (since most individual FireDucks calls return almost instantly, having only built IR rather than computed anything).

There is also a global **benchmark mode** that disables the lazy/batch behavior entirely, executing every method immediately as it's called — useful for apples-to-apples per-call timing comparisons against pandas:

```python
from fireducks.core import get_fireducks_options

get_fireducks_options().set_benchmark_mode(True)
```

## 3.3 Practical consequences of laziness you must code around

**Timing of errors and warnings is different from pandas.** Because computation is deferred, an error caused by an early operation in a chain may not surface until a later materializing call, not at the line that "caused" it. Don't assume a `try/except` placed immediately around the operation that logically causes an error will actually catch it at that point — the exception may only raise when the result is later materialized. Exception *classes* should still match pandas' behavior; exact message text is not guaranteed to match.

**`in` / `__contains__` is an exception — it's eager, not lazy.** `"col" in df` invokes `DataFrame.__contains__()`, which FireDucks implements as a non-lazy method that returns an immediate result rather than adding to the IR plan. Code relying on membership checks (e.g. checking whether a key exists before a lookup) works the way a pandas user would expect, with no special handling needed.


# 4. Pandas Compatibility and Known Differences

FireDucks aims for pandas API compatibility — "you can refer to the pandas documentation to get started" — but explicitly states that *perfect* pandas compatibility is not the goal where it would cost performance. A data consultant generating FireDucks code needs to know the specific, documented places where behavior diverges from real pandas.

## 4.1 `isinstance` checks against `pandas.DataFrame` fail

A FireDucks DataFrame is an instance of a `fireducks.pandas` class, not `pandas.DataFrame`. Code like `isinstance(df, pandas.DataFrame)` returns `False` for a FireDucks object even though it behaves like one. If a script or library does this kind of type check (common in validation code or third-party library internals), it will not recognize a FireDucks object as a DataFrame. Convert with `.to_pandas()` before passing to such code, or write `isinstance` checks against `fireducks.pandas.DataFrame` when the code is FireDucks-aware.

## 4.2 `copy(deep=False)` does not alias data the way pandas does

In real pandas, `df.copy(deep=False)` creates a new DataFrame object that shares the underlying data buffers with the original, so mutating data through one can reflect in the other in certain edge cases. In FireDucks, **shallow copies do not alias underlying data** — metadata-level changes on the "copy" work as expected, but "changes made in the copied instance reflected back into the source instance" do not happen, because FireDucks treats data as immutable internally and allocates new memory whenever an in-place-looking operation is performed. Do not write code that depends on shallow-copy aliasing semantics to propagate a mutation back to a source DataFrame; it will silently not work under FireDucks even though it might have appeared to work in pandas.

## 4.3 Merge/join row ordering can differ from pandas

The row order of a `.merge()` / `.join()` result is not guaranteed to match pandas' row order under FireDucks. If a downstream step depends on merge output being in a particular row order (rather than a particular *set* of rows and values), add an explicit `.sort_values(...)` after the merge rather than relying on incidental pandas ordering — this is a common source of "the numbers are right but the row order differs" discrepancies when migrating a pandas script to FireDucks.

## 4.4 What is explicitly unsupported

- **Internal/private APIs** — anything starting with an underscore that isn't FireDucks' own documented `_evaluate()`.
- **Experimental pandas features** — anything pandas itself marks experimental.
- **DataFrame/Series subclassing** — you cannot subclass FireDucks' DataFrame/Series classes the way some pandas extension libraries do.
- **Custom/extension dtypes** — user-defined pandas extension array types are not supported.
- **Implementation-dependent pandas behavior** — anything that was never a documented pandas guarantee, only an artifact of pandas' specific implementation, is not guaranteed to be replicated.
- **`.apply()` / `.map()` with a user-defined Python function** — not accelerated; see Section 9 for why and what to do instead.

## 4.5 Interoperating with other libraries

Do not mix FireDucks objects and real pandas objects in the same computation and expect it to "just work" — it is not the recommended pattern. When a FireDucks DataFrame needs to go into a library that expects real pandas (plotting libraries, statsmodels, scikit-learn in some code paths, a library that does its own `isinstance(df, pd.DataFrame)` check), convert first:

```python
import fireducks.pandas as pd

fd_df = pd.read_csv("data.csv")
real_pandas_df = fd_df.to_pandas()  # materializes and converts to a genuine pandas.DataFrame

# and the inverse, to bring an existing pandas object into FireDucks:
fd_df2 = pd.from_pandas(real_pandas_df)
```

`.to_pandas()` both materializes any pending lazy computation and produces an object that will pass `isinstance(x, pandas.DataFrame)`.


# 5. Core DataFrame Operations for Data Consulting

Because FireDucks' DataFrame/Series API mirrors pandas directly, the operations below are written exactly as you would in pandas — the only change is the import line (`import fireducks.pandas as pd`). This section exists so the patterns below are retrievable on their own for a RAG system, not because the syntax itself is FireDucks-specific.

## 5.1 Reading and writing data

```python
import fireducks.pandas as pd

df = pd.read_csv("sales.csv", parse_dates=["order_date"])
df = pd.read_parquet("sales.parquet")
df = pd.read_excel("workbook.xlsx", sheet_name="Q3")
df = pd.read_json("records.json")
df = pd.read_sql("SELECT * FROM sales WHERE order_date >= '2026-01-01'", con=connection)

df.to_csv("out.csv", index=False)
df.to_parquet("out.parquet")
df.to_excel("report.xlsx", sheet_name="Summary", index=False)
```

Remember that none of this actually runs until a materializing call happens (Section 3) — building `df` here is cheap; the real cost lands when you first `print(df)`, write it out, or otherwise force evaluation.

## 5.2 Inspecting and profiling a DataFrame

```python
df.shape
df.dtypes
df.head(10)
df.describe()
df.isna().sum()  # null counts per column
df.nunique()  # unique-value counts per column
df.duplicated().sum()  # count of exact duplicate rows
```

Because these calls (especially `.describe()`, `print(df)`) are materializing, expect the first inspection of a freshly loaded/derived DataFrame to be where the actual I/O and upstream computation cost lands, not the lines before it.

## 5.3 Selecting and filtering

```python
df["revenue"]  # always use bracket notation, not df.revenue
df[["customer_id", "revenue"]]
df[df["revenue"] > 100]
df[(df["region"] == "EMEA") & (df["channel"] == "paid")]  # parenthesize each condition
df.loc[df["email"].isna()]
df.query("revenue > 100 and region == 'EMEA'")
```

## 5.4 Adding, transforming, and casting columns

```python
df["profit"] = df["revenue"] - df["cost"]
df["margin"] = df["profit"] / df["revenue"]
df["region"] = df["region"].astype("category")
df["order_date"] = pd.to_datetime(df["order_date"])
df["is_high_value"] = df["revenue"] > df["revenue"].quantile(0.9)
df["grade"] = pd.cut(df["score"], bins=[0, 70, 80, 90, 100], labels=["F", "C", "B", "A"])
```

## 5.5 String and datetime operations

```python
df["name"].str.upper()
df["name"].str.strip()
df["email"].str.contains("@prevaihealth.ai")
df["order_date"].dt.year
df["order_date"].dt.month
df["order_date"].dt.to_period("M")
(df["ship_date"] - df["order_date"]).dt.days
```

## 5.6 Sorting

```python
df.sort_values("revenue", ascending=False)
df.sort_values(["region", "revenue"], ascending=[True, False])
df.nlargest(10, "revenue")
df.nsmallest(10, "revenue")
```


# 6. GroupBy, Aggregation, and Pivot Tables

The `groupby`/`agg`/`pivot_table` API is unchanged from pandas. This is where a data consultant spends most of their time, so it's worth having the full pattern set close at hand.

## 6.1 Basic group_by / agg

```python
df.groupby("region")["revenue"].sum()

df.groupby("region").agg(
    total_revenue=("revenue", "sum"),
    orders=("order_id", "nunique"),
    avg_order_value=("revenue", "mean"),
)

df.groupby(["region", "product_category"])["revenue"].sum()
```

## 6.2 Multiple aggregations per column

```python
df.groupby("customer_id").agg(
    lifetime_value=("revenue", "sum"),
    n_orders=("order_id", "nunique"),
    first_order=("order_date", "min"),
    last_order=("order_date", "max"),
)
```

## 6.3 Resampling / time-bucketed aggregation

```python
monthly = df.set_index("order_date").resample("MS")["revenue"].sum().reset_index()
```

## 6.4 Rolling and expanding windows

```python
df = df.sort_values("order_date")
df["revenue_ma7"] = df["revenue"].rolling(window=7).mean()
df["revenue_cumsum"] = df["revenue"].expanding().sum()
```

## 6.5 Pivot tables (long to wide)

```python
df.pivot_table(values="revenue", index="region", columns="product_category", aggfunc="sum")
```

## 6.6 Group-relative values (the pandas equivalent of a window function)

```python
# Total spend per customer, broadcast back onto every row for that customer
df["customer_ltv"] = df.groupby("customer_id")["revenue"].transform("sum")

# Share of the region's total that this row represents
df["pct_of_region"] = df["revenue"] / df.groupby("region")["revenue"].transform("sum")

# Rank within group
df["rank_in_region"] = df.groupby("region")["revenue"].rank(ascending=False)

# Row change vs. the previous row within a group (e.g. per-customer order-over-order delta)
df = df.sort_values(["customer_id", "order_date"])
df["revenue_change"] = df.groupby("customer_id")["revenue"].diff()
```


# 7. Merging, Joining, and Concatenation

```python
orders.merge(customers, on="customer_id", how="left")
orders.merge(customers, on="customer_id", how="inner")
orders.merge(customers, on="customer_id", how="outer")
orders.merge(customers, left_on="cust_id", right_on="id", how="left")

pd.concat([df1, df2], axis=0)  # stack rows
pd.concat([df1, df2], axis=1)  # side by side columns
```

**Reminder specific to FireDucks (see Section 4.3): merge/join row order is not guaranteed to match pandas.** If downstream logic depends on row order rather than just row content, add an explicit `.sort_values(...)` immediately after the merge/join.


# 8. Missing Data Handling

```python
df.isna().sum()
df.dropna()
df.dropna(subset=["customer_id", "order_date"])
df["revenue"].fillna(0)
df["category"].fillna("Unknown")
df["revenue"].fillna(df["revenue"].mean())
df["value"].interpolate()
```


# 9. Performance Tuning and Avoiding Fallback

FireDucks' entire value proposition depends on staying on the "compiled/accelerated" code path. When an operation isn't supported by the compiler, FireDucks silently **falls back**: it converts the FireDucks data structure to real pandas, runs the pandas method, and converts the result back to a FireDucks structure. This round-trip conversion has real cost and gives you *no* acceleration for that operation — it's the single biggest reason a "FireDucks-accelerated" script can end up no faster than plain pandas.

## 9.1 Never pass a user-defined function to `.apply()`

```python
# WRONG — falls back to pandas, no acceleration, and is a common mistake in generated code
df["flag"] = df.apply(lambda row: row["a"] > 2, axis=1)

# RIGHT — vectorized, stays on the compiled path
df["flag"] = df["a"] > 2
```

FireDucks' current optimizer generates and compiles an intermediate language for known DataFrame operations; it cannot compile an arbitrary Python callable, so `.apply()`/`.map()` with a user function always falls back. When generating FireDucks code, always look for a vectorized equivalent of what a `.apply()` call is trying to do before reaching for `.apply()`.

## 9.2 Never write Python row-loops over a DataFrame

```python
# WRONG — slow in pandas AND falls back / defeats FireDucks entirely
s = 0
for i in range(len(df)):
    if df["A"][i] > 2:
        s += df["B"][i]

# RIGHT
s = df[df["A"] > 2]["B"].sum()
```

## 9.3 Always use bracket notation for column access

```python
df["A"]  # correct — always works, always compiler-friendly
df.A  # avoid — attribute-style access can collide with DataFrame methods/attributes
# of the same name and can disable compiler optimizations
```

## 9.4 Avoid pandas' ambiguous/undefined-behavior indexing patterns

```python
# AVOID — ambiguous even in real pandas (unclear if this returns a view or a copy)
df["A"][1] = 2

# PREFER — explicit, unambiguous positional assignment
df.iloc[1, df.columns.get_loc("A")] = 2
```

## 9.5 Diagnosing and logging fallback

Enable fallback logging to see, at runtime, which operations silently dropped to pandas:

```bash
FIREDUCKS_FLAGS="-Wfallback" python3 your_script.py
```

For deeper performance debugging, enable tracing, which writes a `trace.json` you can inspect:

```bash
FIREDUCKS_FLAGS="--trace=3" python3 your_script.py
```

## 9.6 Profiling in Jupyter (FireDucks 0.10.9+)

```python
%load_ext fireducks.ipyext
```
```python
%%fireducks.profile
# your DataFrame code here — do not put import statements in this cell,
# import fireducks/pandas in a separate cell first
```


# 10. FireDucks' Own API Extensions

Beyond the pandas-compatible surface, FireDucks exposes a small number of its own functions and controls.

## 10.1 Conversion to and from real pandas

```python
import fireducks.pandas as pd

fd_df = pd.read_csv("data.csv")
real_df = fd_df.to_pandas()  # materialize + convert to a genuine pandas.DataFrame
fd_df2 = pd.from_pandas(real_df)  # bring an existing pandas object into FireDucks
```

## 10.2 Explicit evaluation and benchmark mode

```python
df = pd.read_csv("data.csv")._evaluate()  # force materialization now, for isolated timing

from fireducks.core import get_fireducks_options

get_fireducks_options().set_benchmark_mode(True)  # disable batching; execute eagerly for timing
```

## 10.3 Feature-generation helpers

FireDucks ships a small number of pre-optimized helpers aimed at feature-engineering workloads (e.g. `fireducks.pandas.aggregate` and `fireducks.pandas.multi_target_encoding`) — these are FireDucks-specific conveniences, not pandas API surface, intended for ML feature-generation pipelines rather than general data consulting. Reach for the standard `.groupby().agg()` patterns in Section 6 for general analytics work; only consider these when the specific task is bulk feature/target encoding for a modeling pipeline.


# 11. Statistical Analysis Patterns

Since the DataFrame API is pandas-compatible, the same statistical patterns a consultant uses in pandas apply directly. As with any FireDucks call, remember these materialize lazily — nothing computes until you inspect/use the result.

## 11.1 Descriptive statistics

```python
df.describe()
df["revenue"].mean()
df["revenue"].median()
df["revenue"].std()
df["revenue"].quantile([0.25, 0.5, 0.75])
df["revenue"].skew()
df["revenue"].kurt()
```

## 11.2 Correlation

```python
df[["price", "units_sold", "revenue"]].corr()
df["price"].corr(df["units_sold"])
```

## 11.3 Outlier detection

```python
q1, q3 = df["revenue"].quantile(0.25), df["revenue"].quantile(0.75)
iqr = q3 - q1
outliers = df[(df["revenue"] < q1 - 1.5 * iqr) | (df["revenue"] > q3 + 1.5 * iqr)]

z = (df["revenue"] - df["revenue"].mean()) / df["revenue"].std()
outliers_z = df[z.abs() > 3]
```

## 11.4 Hypothesis testing and regression (hand off to scipy/statsmodels)

Statistical test/regression libraries generally expect real numpy arrays or real pandas objects, not FireDucks-lazy structures — materialize with `.to_numpy()` or `.to_pandas()` first:

```python
from scipy import stats
import numpy as np

group_a = df[df["variant"] == "A"]["conversion_rate"].to_numpy()
group_b = df[df["variant"] == "B"]["conversion_rate"].to_numpy()
t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

x = df["marketing_spend"].to_numpy()
y = df["revenue"].to_numpy()
slope, intercept = np.polyfit(x, y, 1)
```


# 12. Data Consulting Workflows

These templates mirror how an analyst should approach a client request; the syntax is FireDucks/pandas throughout.

## 12.1 Workflow: Profile a new/unfamiliar dataset

Always run this before trusting any number from data you haven't seen before.

```python
import fireducks.pandas as pd

df = pd.read_csv("client_data.csv")
print(df.shape)
print(df.dtypes)
print(df.head(10))
print(df.describe())
print(df.isna().sum())
for col in df.columns:
    print(col, df[col].nunique(), "unique values")
print("duplicate rows:", df.duplicated().sum())
dup_keys = df.groupby("id_column").size()
print("duplicate keys:", (dup_keys > 1).sum())
```

Report back: row/column counts, dtypes, null rates per column, duplicate-key issues, and the date range covered, before doing any analysis on top of the data.

## 12.2 Workflow: Diagnose a metric drop / "why did X change"

```python
current = df[df["week"] == current_week]
prior = df[df["week"] == prior_week]


def kpis(frame):
    return pd.Series(
        {
            "revenue": frame["revenue"].sum(),
            "orders": frame["order_id"].nunique(),
            "aov": frame["revenue"].sum() / frame["order_id"].nunique(),
        }
    )


print("current:", kpis(current))
print("prior:", kpis(prior))

decomp = (
    df[df["week"].isin([current_week, prior_week])]
    .groupby(["segment", "week"])["revenue"]
    .sum()
    .unstack("week")
)
decomp["delta"] = decomp[current_week] - decomp[prior_week]
decomp = decomp.sort_values("delta")
print(decomp)
```

Decompose by every available dimension (region, channel, segment) to find where a change concentrates, and check for mix-shift (did segment sizes change even if per-segment behavior didn't) before concluding the cause.

## 12.3 Workflow: Segment / cohort comparison

```python
segment_summary = df.groupby("segment").agg(
    customers=("customer_id", "nunique"),
    total_revenue=("revenue", "sum"),
    avg_order_value=("revenue", "mean"),
)
segment_summary["revenue_per_customer"] = (
    segment_summary["total_revenue"] / segment_summary["customers"]
)
segment_summary["pct_of_total"] = (
    segment_summary["total_revenue"] / segment_summary["total_revenue"].sum() * 100
).round(1)
segment_summary = segment_summary.sort_values("total_revenue", ascending=False)
```

## 12.4 Workflow: Build a KPI / executive summary table

```python
kpis = pd.Series(
    {
        "total_revenue": df["revenue"].sum(),
        "total_orders": df["order_id"].nunique(),
        "total_customers": df["customer_id"].nunique(),
    }
)
kpis["avg_order_value"] = round(kpis["total_revenue"] / kpis["total_orders"], 2)
kpis["revenue_per_customer"] = round(kpis["total_revenue"] / kpis["total_customers"], 2)
```

Present this as a short narrative — total revenue, order count, customer count, average order value, and the period-over-period change — not as a raw printed Series.

## 12.5 Workflow: Data quality / anomaly sweep before trusting a number

```python
issues = []
if df["order_date"].isna().sum() > 0:
    issues.append("missing order dates")
if (df["revenue"] < 0).sum() > 0:
    issues.append("negative revenue rows")
dup_keys = df.groupby("order_id").size()
if (dup_keys > 1).sum() > 0:
    issues.append(f"{(dup_keys > 1).sum()} duplicated order_id values")
if (df["order_date"] > pd.Timestamp.today()).sum() > 0:
    issues.append("rows with future order dates")
print("Data quality issues found:", issues or "none")
```

## 12.6 How to present results (report structure)

Structure the written output as: headline finding in one sentence, supporting numbers in a small table, one chart if it clarifies a trend or comparison, caveats/data-quality notes, and a recommended next step the numbers imply. Never hand back raw `df.describe()` output as the final answer — synthesize it into plain-language sentences with the specific numbers embedded.


# 13. Charting and Visualization

**Important FireDucks-specific note:** most plotting libraries (matplotlib, seaborn) and some statistics libraries expect a genuine `pandas.DataFrame`/`Series`, and FireDucks objects fail `isinstance(x, pandas.DataFrame)` checks (Section 4.1). If a chart call behaves oddly or a library errors on a FireDucks object, convert first with `.to_pandas()`.

## 13.1 Matplotlib (static, report-ready charts)

```python
import matplotlib.pyplot as plt

monthly = df.set_index("order_date").resample("MS")["revenue"].sum().reset_index()
pdf = monthly.to_pandas()  # safest: hand matplotlib a real pandas object

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(pdf["order_date"], pdf["revenue"], marker="o")
ax.set_title("Monthly Revenue Trend")
ax.set_ylabel("Revenue ($)")
fig.tight_layout()
fig.savefig("monthly_revenue.png", dpi=150)
```

```python
summary = df.groupby("region")["revenue"].sum().sort_values(ascending=False).to_pandas()
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(summary.index, summary.values, color="#4C72B0")
ax.set_title("Revenue by Region")
plt.xticks(rotation=30, ha="right")
fig.tight_layout()
fig.savefig("revenue_by_region.png", dpi=150)
```

## 13.2 Plotly (interactive HTML deliverables)

```python
import plotly.express as px

pdf = df.groupby("category")["revenue"].sum().sort_values(ascending=False).reset_index().to_pandas()
fig = px.bar(pdf, x="category", y="revenue", title="Revenue by Category")
fig.write_html("revenue_by_category.html")
```

## 13.3 Chart-type selection guide

| Question being answered | Chart type |
|---|---|
| How does a metric trend over time? | Line chart |
| How do categories compare on one metric? | Bar chart, sorted by value |
| How is a metric distributed? | Histogram / box plot |
| Is there a relationship between two numeric variables? | Scatter plot |
| How does a whole break into parts? | Stacked bar (avoid pie charts beyond ~4-5 slices) |


# 14. End-to-End Worked Case Study

**Scenario**: "Revenue looked flat last quarter versus last year. What's going on?" Data: `orders.csv` with columns `order_id, customer_id, order_date, region, channel, product_category, revenue, cost`.

```python
import fireducks.pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("orders.csv", parse_dates=["order_date"])

# 1. Profile
print(df.shape, df.dtypes)

# 2. Define comparison windows
this_q = (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-06-30"))
last_q_ly = (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-30"))


def window_kpis(frame, start, end):
    w = frame[(frame["order_date"] >= start) & (frame["order_date"] <= end)]
    return pd.Series(
        {
            "revenue": w["revenue"].sum(),
            "orders": w["order_id"].nunique(),
            "customers": w["customer_id"].nunique(),
        }
    )


cur = window_kpis(df, *this_q)
prior = window_kpis(df, *last_q_ly)
print("This Q:", cur)
print("Same Q last year:", prior)

# 3. Decompose by region/channel to find where flatness concentrates
mask = ((df["order_date"] >= this_q[0]) & (df["order_date"] <= this_q[1])) | (
    (df["order_date"] >= last_q_ly[0]) & (df["order_date"] <= last_q_ly[1])
)
sub = df[mask].copy()
sub["period"] = (sub["order_date"] >= this_q[0]).map({True: "this_q", False: "last_q_ly"})

decomp = sub.groupby(["region", "channel", "period"])["revenue"].sum().unstack("period")
decomp["yoy_pct"] = ((decomp["this_q"] - decomp["last_q_ly"]) / decomp["last_q_ly"] * 100).round(1)
decomp = decomp.sort_values("yoy_pct")
print(decomp)

# 4. Monthly trend chart for the narrative
monthly = df.set_index("order_date").resample("MS")["revenue"].sum().reset_index().to_pandas()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(monthly["order_date"], monthly["revenue"], marker="o")
ax.set_title("Monthly Revenue Trend")
fig.tight_layout()
fig.savefig("monthly_revenue_trend.png", dpi=150)

# 5. Narrative for the client:
# "Total revenue was flat (+0.4% YoY) this quarter, but that hides a split: APAC paid
#  declined 18% YoY while NA organic grew 22% YoY, offsetting it. The APAC paid decline
#  concentrates in the last six weeks and coincides with a drop in unique customers, not
#  order value -- suggesting an acquisition/traffic issue rather than pricing or basket size."
```


# 15. Common Pitfalls and Anti-Patterns

- **Passing a UDF to `.apply()`/`.map()` expecting acceleration.** It silently falls back to real pandas — no speedup, only conversion overhead. Use vectorized operations (Section 9.1).
- **Assuming shallow copies alias data like pandas.** `df.copy(deep=False)` does not propagate mutations back to the source under FireDucks (Section 4.2).
- **Trusting merge/join row order.** Add an explicit `.sort_values()` after any merge/join whose downstream logic depends on row order (Section 4.3).
- **`isinstance(df, pandas.DataFrame)` checks.** These fail for FireDucks objects; convert with `.to_pandas()` first, or check against `fireducks.pandas.DataFrame`.
- **Assuming a `try/except` around a chain catches the error at the "logical" line.** Because of lazy evaluation, the exception may only raise when a later materializing call runs.
- **Using `df.column_name` instead of `df["column_name"]`.** Attribute access can disable compiler optimizations and is ambiguous with DataFrame methods.
- **Writing Python row-loops over a DataFrame.** Always slower than a vectorized filter/aggregation, and defeats FireDucks' entire purpose.
- **Feeding a FireDucks object straight into matplotlib/seaborn/statsmodels without checking.** These libraries may expect real pandas; call `.to_pandas()` first if anything behaves oddly.
- **Expecting Windows support.** FireDucks has no native Windows build; WSL is the workaround.
- **Subclassing FireDucks DataFrame/Series, or relying on pandas extension dtypes.** Both are explicitly unsupported.


# 16. Quick-Reference Cheat Sheet

```python
# --- Setup ---
import fireducks.pandas as pd  # explicit import (preferred for new code)
# python3 -m fireducks.pandas script.py        # import hook, for existing scripts
# %load_ext fireducks.pandas                   # Jupyter magic, for existing notebooks

# --- I/O ---
pd.read_csv(path)
pd.read_parquet(path)
pd.read_excel(path)
pd.read_sql(query, con)
df.to_csv(path, index=False)
df.to_parquet(path)

# --- Inspect (materializes) ---
df.shape
df.dtypes
df.head()
df.describe()
df.isna().sum()

# --- Select / filter (bracket notation only) ---
df["col"]
df[(df["a"] > 1) & (df["b"].isin([...]))]

# --- Transform ---
df["c"] = df["a"] * df["b"]
df["d"] = df["x"].astype("category")

# --- Aggregate ---
df.groupby("key").agg(total=("value", "sum"))
df["group_total"] = df.groupby("key")["value"].transform("sum")  # window-function equivalent

# --- Merge / reshape ---
a.merge(b, on="key", how="left").sort_values("key")  # always re-sort after merge if order matters
df.pivot_table(values="v", index="i", columns="c", aggfunc="sum")

# --- Performance rules ---
# Never: df.apply(lambda row: ..., axis=1)   -> falls back to pandas, no acceleration
# Never: for i in range(len(df)): ...        -> defeats FireDucks entirely
# Always: df["col"] not df.col
# FIREDUCKS_FLAGS="-Wfallback" python3 script.py   # log fallback events
# FIREDUCKS_FLAGS="--trace=3" python3 script.py    # detailed perf trace -> trace.json

# --- FireDucks-specific ---
df._evaluate()  # force materialization now (benchmarking/debugging)
df.to_pandas()  # materialize + convert to a genuine pandas.DataFrame
pd.from_pandas(real_df)  # bring a real pandas object into FireDucks

# --- Known incompatibilities with pandas ---
# isinstance(fd_df, pandas.DataFrame) -> False
# df.copy(deep=False) does not alias data back to the source
# merge/join row order not guaranteed to match pandas
# try/except around a chain may not catch at the "logical" line (lazy evaluation)
```


# 17. FAQ, Licensing, and Support

**Internal data format:** FireDucks' multi-threaded CPU backend is built on Apache Arrow.

**License:** 3-Clause BSD License.

**Platforms:** Linux (manylinux, x86_64) and macOS (ARM). No native Windows build; use WSL.

**Commercial support:** none, as of this writing — community support happens via a public Slack workspace and the GitHub repository (`fireducks-dev/fireducks`).

**Reporting bugs / feature requests:** via GitHub issues, or `contact@fireducks.jp.nec.com`.

**Staying current:** watch PyPI release notifications for the `fireducks` package, or follow `@fireducksdev`. This document is pinned to FireDucks **1.4.4** (December 2025) — re-verify version-specific behavior (especially anything under Section 9's `%%fireducks.profile`, which requires 0.10.9+) against the official docs at `fireducks-dev.github.io/docs` if the installed version is materially newer.

*End of reference. When in doubt: it's pandas syntax, it's lazy under the hood, avoid `.apply()` with custom functions, always use bracket notation, and convert to a real pandas object with `.to_pandas()` before handing a FireDucks object to another library.*
