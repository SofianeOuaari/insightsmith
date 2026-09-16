# Pandas recipes

Worked answers to the shapes a data question comes in. Find the shape that matches, then adapt the column names. Placeholder names like `category`, `measure` and `date_column` are stand-ins for real columns from the card.

## 1. Total of a measure by category

Questions like: total revenue by region; sum of sales per product type; how much did each category generate.

```python
result = (
    df.groupby("category", as_index=False)["measure"]
    .sum()
    .rename(columns={"measure": "total_measure"})
    .sort_values("total_measure", ascending=False)
)
```

## 2. Average of a measure by category

Questions like: average salary by department; mean quality of sleep per occupation; what is the typical value for each group.

```python
result = (
    df.groupby("category")
    .agg(mean_measure=("measure", "mean"), n=("measure", "size"))
    .reset_index()
    .sort_values("mean_measure", ascending=False)
)
```

**Watch out.** Carry the group size. A mean over one row is that row, and without `n` in the result nobody downstream can tell.

## 3. Ratio of two measures by category

Questions like: ratio of balance to credit limit by grade; margin by product; conversion rate per channel; which category has the highest ratio of x to y.

```python
result = (
    df.groupby("category")
    .agg(num=("numerator", "sum"), den=("denominator", "sum"), n=("numerator", "size"))
    .reset_index()
    .query("den > 0")
    .assign(ratio=lambda d: d["num"] / d["den"])
    .sort_values("ratio", ascending=False)
)
```

**Watch out.** Two different questions hide here. The ratio of the sums weights big rows heavily; the mean of the per-row ratios treats every row alike, and they often rank the groups differently. The code above is the ratio of sums, which is usually what 'ratio of X to Y' means for a group. Keep `n` and guard the zero denominator either way.

## 4. Top N rows by a measure

Questions like: top 10 customers by spend; which five products sold most; highest revenue rows.

```python
result = df.nlargest(10, "measure")
```

## 5. Correlation between two numeric columns

Questions like: is there a correlation between price and quantity; how strongly does spend relate to conversions; relationship between two measures.

```python
result = pd.DataFrame(
    {
        "pearson": [df["first_column"].corr(df["second_column"])],
        "spearman": [df["first_column"].corr(df["second_column"], method="spearman")],
    }
)
```

**Watch out.** `df.corr()` returns a whole matrix, not one number, and `.item()` on it raises. Ask for the pair you mean. Reporting Spearman beside Pearson costs nothing and shows immediately when outliers or a curve are driving the result.

## 6. Counts and shares of a category

Questions like: distribution of loan status; how many of each category; what share of customers are in each segment.

```python
result = (
    df["category"]
    .value_counts()
    .rename_axis("category")
    .reset_index(name="n")
    .assign(pct=lambda d: (d["n"] / d["n"].sum() * 100).round(1))
)
```

## 7. A measure over time

Questions like: monthly revenue trend; how have sales changed over the year; revenue by quarter.

```python
result = (
    df.assign(month=pd.to_datetime(df["date_column"]).dt.to_period("M").dt.to_timestamp())
    .groupby("month", as_index=False)["measure"]
    .sum()
    .sort_values("month")
)
```

**Watch out.** Sort by the time column, not by the measure. A trend drawn in group order is a scribble. If the dates are text, parse them first.

## 8. Compare a measure across two groups

Questions like: do churned customers spend more; difference between treatment and control; compare average by gender.

```python
result = (
    df.groupby("group_column")["measure"]
    .agg(n="size", mean_measure="mean", median_measure="median", std_measure="std")
    .reset_index()
)
```

**Watch out.** A difference between two group averages can reverse once you split by a third column that differs between them. If such a column exists, group by both before concluding anything about the first.

## 9. Distinct count per category

Questions like: how many distinct products per state; unique customers by region; number of different values in each group.

```python
result = (
    df.groupby("category", as_index=False)["id_column"]
    .nunique()
    .rename(columns={"id_column": "distinct_ids"})
    .sort_values("distinct_ids", ascending=False)
)
```

## 10. Filter, then aggregate

Questions like: average order value for customers in EMEA; total revenue where the channel is online; how many rows match a condition.

```python
subset = df[df["category"] == "the value"]
result = pd.DataFrame({"n": [len(subset)], "total_measure": [subset["measure"].sum()]})
```

**Watch out.** Report how many rows survived the filter. A total over three rows and a total over three thousand look identical otherwise.

## 11. Share of a total by category

Questions like: what percentage of revenue comes from each region; share of total by product; which category accounts for most of the total.

```python
result = (
    df.groupby("category", as_index=False)["measure"]
    .sum()
    .rename(columns={"measure": "total_measure"})
    .assign(pct_of_total=lambda d: (d["total_measure"] / d["total_measure"].sum() * 100).round(1))
    .sort_values("pct_of_total", ascending=False)
)
```

## 12. Missing values by column

Questions like: how many nulls are there; which columns have missing data; data quality summary.

```python
nulls = df.isna().sum()
result = (
    nulls[nulls > 0]
    .rename_axis("column")
    .reset_index(name="nulls")
    .assign(pct=lambda d: (d["nulls"] / len(df) * 100).round(1))
    .sort_values("nulls", ascending=False)
)
```

**Watch out.** Missingness is rarely random. If the rate differs sharply between groups, the rows that remain are a biased sample and any average over them is biased too.

## 13. Outliers in a numeric column

Questions like: find the outliers in amount; which rows are unusually large; detect anomalies in a measure.

```python
q1, q3 = df["measure"].quantile([0.25, 0.75])
iqr = q3 - q1
result = df[(df["measure"] < q1 - 1.5 * iqr) | (df["measure"] > q3 + 1.5 * iqr)].sort_values(
    "measure", ascending=False
)
```

**Watch out.** The IQR fence collapses when the middle half of the column is a single value, and then everything looks like an outlier. Sanity-check the count against the number of rows.

## 14. Cross-tabulation of two categories

Questions like: revenue by region and channel; breakdown across two dimensions; pivot category against category.

```python
result = df.pivot_table(
    values="measure",
    index="row_category",
    columns="column_category",
    aggfunc="sum",
    fill_value=0,
).reset_index()
```

## 15. Period over period change

Questions like: year over year growth; how did this quarter compare with last; change since the previous period.

```python
periods = (
    df.assign(period=pd.to_datetime(df["date_column"]).dt.to_period("M").dt.to_timestamp())
    .groupby("period", as_index=False)["measure"]
    .sum()
    .rename(columns={"measure": "total_measure"})
    .sort_values("period")
)
result = periods.assign(pct_change=lambda d: (d["total_measure"].pct_change() * 100).round(1))
```

**Watch out.** A percentage change against a near-zero base is enormous and means nothing. Keep the totals in the result so the reader can see the base.

## 16. Rate of a yes or no outcome by category

Questions like: churn rate by region; what percentage of customers in each segment defaulted; how does the readmission rate vary across departments; likelihood of recommending the service by channel.

```python
grouped = df.groupby("category")["flag_column"]
result = grouped.agg(
    rate_pct=lambda s: round(s.astype(bool).mean() * 100, 1), n="size"
).reset_index()
result = result[result["n"] >= 30].sort_values("rate_pct", ascending=False)
```

**Watch out.** A rate is a mean of a 0/1 column, not a count — counting the yes rows gives the volume, not the rate, and ranks big groups top every time. Carry `n`: a 100% rate over three rows is noise, which is why the small groups are filtered out rather than shown and explained away.

## 17. Top N groups by an aggregate

Questions like: which cities have the highest average price; which states have the highest average sales; which five departments have the longest waiting time; which product types earn the most per order.

```python
result = (
    df.groupby("category").agg(avg_measure=("measure", "mean"), n=("measure", "size")).reset_index()
)
result = (
    result[result["n"] >= 30]
    .assign(avg_measure=lambda d: d["avg_measure"].round(2))
    .sort_values("avg_measure", ascending=False)
    .head(10)
)
```

**Watch out.** This is not `sort().head()` on the rows — that returns the ten largest *rows*, and the question asked which groups rank highest on an average. Aggregate first, then sort. Without the size filter the top of the list is whichever group happens to have one member.

## 18. A measure by a numeric column cut into bands

Questions like: average satisfaction by age group; how does satisfaction vary by household income; default rate by tenure band; does the effect change across income brackets.

```python
edges = [-float("inf"), 25, 40, 60, float("inf")]
bands = pd.cut(df["numeric_column"], bins=edges, labels=["<25", "25-39", "40-59", "60+"])
result = (
    df.assign(band=bands)
    .groupby("band", as_index=False, observed=True)
    .agg(avg_measure=("measure", "mean"), n=("measure", "size"))
)
result["avg_measure"] = result["avg_measure"].round(2)
```

**Watch out.** Grouping on the raw numeric column makes one group per distinct value — hundreds of groups of one row. Bands are a choice, not a fact: state the edges in the answer, because moving them can move the conclusion. Sort by the band, never by the value, or the reader cannot read the trend.

## 19. Distribution of a numeric column

Questions like: what is the distribution of annual salaries; distribution of patient satisfaction scores; how are prices spread out; what does the spread of subscriber counts look like.

```python
stats = df["numeric_column"].describe(percentiles=[0.25, 0.5, 0.75])
result = stats.round(2).to_frame().T.reset_index(drop=True)
```

**Watch out.** The distribution of a *numeric* column is quantiles, not `value_counts` — counting distinct salaries yields one row per employee. Report the median beside the mean: where they part company the column is skewed and the mean alone will mislead.

## 20. A measure by two categories at once

Questions like: how do impressions vary by channel and quarter; average lift by segment for each metric; sales by region and product type.

```python
result = (
    df.groupby(["first_category", "second_category"], as_index=False, observed=True)
    .agg(avg_measure=("measure", "mean"), n=("measure", "size"))
    .sort_values(["first_category", "second_category"])
)
result["avg_measure"] = result["avg_measure"].round(2)
```

**Watch out.** Two keys multiply: ten channels by twelve quarters is 120 rows, many of them thin. Keep the result long rather than pivoting it wide — a long frame charts and reads back correctly, where a wide one hides the group sizes. Averaging the per-cell averages afterwards is not the overall average unless every cell holds the same number of rows.

## 21. Most common value per category

Questions like: most common waiting time across departments; which product each region buys most; the typical sleep disorder per occupation.

```python
counts = (
    df.groupby(["category", "value_column"], observed=True)
    .size()
    .reset_index(name="n")
    .sort_values(["category", "n"], ascending=[True, False])
)
result = counts.groupby("category", as_index=False, observed=True).first()
```

**Watch out.** The mode is not the average, and `.mode()` returns every tied value, so a flat distribution yields several rows per group. Keep `n` in the result: a mode of 4 out of 400 is not what a reader hears in 'most common'.

## 22. Does one column affect another

Questions like: how does physical activity level affect stress; how does age affect the likelihood of readmission; does contract type influence churn; what is the impact of occupation on sleep quality; how does the use of AI affect audit effectiveness.

```python
# One question, two shapes — pick by the dtype of the column doing the affecting.
# Categorical driver: compare the outcome across its values.
result = (
    df.groupby("driver_column", as_index=False, observed=True)
    .agg(avg_outcome=("outcome_column", "mean"), n=("outcome_column", "size"))
    .sort_values("avg_outcome", ascending=False)
)
result["avg_outcome"] = result["avg_outcome"].round(2)
# Numeric driver, instead: correlate, or band it first with recipe 18.
# result = pd.DataFrame({"pearson": [df["driver_column"].corr(df["outcome_column"]).round(3)]})
```

**Watch out.** 'Affects' is a causal word and none of this establishes cause. Answer with the difference and let the reader draw the arrow: say group A averages 12 against group B's 9, never that A *causes* the 3. A difference across groups can also reverse once a third column is held fixed, so a large gap is a finding to check, not a conclusion.

## 23. Whether a difference between two groups is real

Questions like: which segments show statistically significant improvements; is the difference between the variant and control real; is that gap significant.

```python
summary = (
    df.groupby("group_column", observed=True)["measure"]
    .agg(mean="mean", sd="std", n="size")
    .reset_index()
)
a, b = summary.iloc[0], summary.iloc[1]
se = (a["sd"] ** 2 / a["n"] + b["sd"] ** 2 / b["n"]) ** 0.5
diff = a["mean"] - b["mean"]
result = summary.assign(
    difference=round(diff, 3),
    ci_low=round(diff - 1.96 * se, 3),
    ci_high=round(diff + 1.96 * se, 3),
)
```

**Watch out.** Report the interval, not a verdict. An interval straddling zero means the data does not settle the question — it is not evidence the groups are the same. Testing every segment until one looks significant is how noise gets published: with twenty segments, one crossing the line at the usual threshold is the expected result when nothing is going on. Say how many were examined, and keep the group sizes in the result.
