"""Generate the coder's recipe books, one per dataframe API.

The bundled guides are API reference: they answer "what does group_by do". A
small model's problem is not knowing the API, it is *composing* one. Measured on
this project, a 1.3B coder answered ``{"code": "SUCCESS"}`` and an 8B one needed
retries on anything shaped like a ratio. Both are easier to help with a worked
example of the whole question than with the documentation for one method.

So this is a catalogue of the shapes a data question actually comes in, each
with the snippet that answers it end to end, and the trap that shape carries
where there is one. Retrieval is the same BM25 used for the guides, which is why
the output is Markdown with the same heading convention.

One list, two renderings. Keeping the analytic patterns in a single place is
what stops the Polars and pandas books drifting into different catalogues::

    python scripts/build_recipes.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "insightsmith" / "knowledge"


@dataclass(frozen=True, slots=True)
class Recipe:
    """One shape of question, and the code that answers it in each API."""

    title: str
    #: Phrasings a reader might actually use. These are what retrieval matches,
    #: so they carry the question's vocabulary rather than the API's.
    asks: tuple[str, ...]
    polars: str
    pandas: str
    #: The mistake this shape invites. Omitted where there isn't one.
    trap: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)


RECIPES: Final[tuple[Recipe, ...]] = (
    Recipe(
        title="Total of a measure by category",
        asks=(
            "total revenue by region",
            "sum of sales per product type",
            "how much did each category generate",
        ),
        polars="""result = (
    df.group_by("category")
      .agg(pl.col("measure").sum().alias("total_measure"))
      .sort("total_measure", descending=True)
)""",
        pandas="""result = (
    df.groupby("category", as_index=False)["measure"]
      .sum()
      .rename(columns={"measure": "total_measure"})
      .sort_values("total_measure", ascending=False)
)""",
    ),
    Recipe(
        title="Average of a measure by category",
        asks=(
            "average salary by department",
            "mean quality of sleep per occupation",
            "what is the typical value for each group",
        ),
        polars="""result = (
    df.group_by("category")
      .agg(
          pl.col("measure").mean().alias("mean_measure"),
          pl.len().alias("n"),
      )
      .sort("mean_measure", descending=True)
)""",
        pandas="""result = (
    df.groupby("category")
      .agg(mean_measure=("measure", "mean"), n=("measure", "size"))
      .reset_index()
      .sort_values("mean_measure", ascending=False)
)""",
        trap=(
            "Carry the group size. A mean over one row is that row, and without "
            "`n` in the result nobody downstream can tell."
        ),
    ),
    Recipe(
        title="Ratio of two measures by category",
        asks=(
            "ratio of balance to credit limit by grade",
            "margin by product",
            "conversion rate per channel",
            "which category has the highest ratio of x to y",
        ),
        polars="""result = (
    df.group_by("category")
      .agg(
          pl.col("numerator").sum().alias("num"),
          pl.col("denominator").sum().alias("den"),
          pl.len().alias("n"),
      )
      .filter(pl.col("den") > 0)
      .with_columns((pl.col("num") / pl.col("den")).alias("ratio"))
      .sort("ratio", descending=True)
)""",
        pandas="""result = (
    df.groupby("category")
      .agg(num=("numerator", "sum"), den=("denominator", "sum"), n=("numerator", "size"))
      .reset_index()
      .query("den > 0")
      .assign(ratio=lambda d: d["num"] / d["den"])
      .sort_values("ratio", ascending=False)
)""",
        trap=(
            "Two different questions hide here. The ratio of the sums weights big "
            "rows heavily; the mean of the per-row ratios treats every row alike, "
            "and they often rank the groups differently. The code above is the "
            "ratio of sums, which is usually what 'ratio of X to Y' means for a "
            "group. Keep `n` and guard the zero denominator either way."
        ),
    ),
    Recipe(
        title="Top N rows by a measure",
        asks=(
            "top 10 customers by spend",
            "which five products sold most",
            "highest revenue rows",
        ),
        polars="""result = df.sort("measure", descending=True).head(10)""",
        pandas="""result = df.nlargest(10, "measure")""",
    ),
    Recipe(
        title="Correlation between two numeric columns",
        asks=(
            "is there a correlation between price and quantity",
            "how strongly does spend relate to conversions",
            "relationship between two measures",
        ),
        polars="""result = df.select(
    pl.corr("first_column", "second_column").alias("pearson"),
    pl.corr("first_column", "second_column", method="spearman").alias("spearman"),
)""",
        pandas="""result = pd.DataFrame({
    "pearson": [df["first_column"].corr(df["second_column"])],
    "spearman": [df["first_column"].corr(df["second_column"], method="spearman")],
})""",
        trap=(
            "`df.corr()` returns a whole matrix, not one number, and `.item()` on "
            "it raises. Ask for the pair you mean. Reporting Spearman beside "
            "Pearson costs nothing and shows immediately when outliers or a curve "
            "are driving the result."
        ),
    ),
    Recipe(
        title="Counts and shares of a category",
        asks=(
            "distribution of loan status",
            "how many of each category",
            "what share of customers are in each segment",
        ),
        polars="""result = (
    df.group_by("category")
      .agg(pl.len().alias("n"))
      .with_columns((pl.col("n") / pl.col("n").sum() * 100).round(1).alias("pct"))
      .sort("n", descending=True)
)""",
        pandas="""result = (
    df["category"].value_counts()
      .rename_axis("category").reset_index(name="n")
      .assign(pct=lambda d: (d["n"] / d["n"].sum() * 100).round(1))
)""",
    ),
    Recipe(
        title="A measure over time",
        asks=(
            "monthly revenue trend",
            "how have sales changed over the year",
            "revenue by quarter",
        ),
        polars="""result = (
    df.with_columns(pl.col("date_column").dt.truncate("1mo").alias("month"))
      .group_by("month")
      .agg(pl.col("measure").sum().alias("total_measure"))
      .sort("month")
)""",
        pandas="""result = (
    df.assign(month=pd.to_datetime(df["date_column"]).dt.to_period("M").dt.to_timestamp())
      .groupby("month", as_index=False)["measure"].sum()
      .sort_values("month")
)""",
        trap=(
            "Sort by the time column, not by the measure. A trend drawn in "
            "group order is a scribble. If the dates are text, parse them first."
        ),
    ),
    Recipe(
        title="Compare a measure across two groups",
        asks=(
            "do churned customers spend more",
            "difference between treatment and control",
            "compare average by gender",
        ),
        polars="""result = (
    df.group_by("group_column")
      .agg(
          pl.len().alias("n"),
          pl.col("measure").mean().alias("mean_measure"),
          pl.col("measure").median().alias("median_measure"),
          pl.col("measure").std().alias("std_measure"),
      )
      .sort("group_column")
)""",
        pandas="""result = (
    df.groupby("group_column")["measure"]
      .agg(n="size", mean_measure="mean", median_measure="median", std_measure="std")
      .reset_index()
)""",
        trap=(
            "A difference between two group averages can reverse once you split "
            "by a third column that differs between them. If such a column exists, "
            "group by both before concluding anything about the first."
        ),
    ),
    Recipe(
        title="Distinct count per category",
        asks=(
            "how many distinct products per state",
            "unique customers by region",
            "number of different values in each group",
        ),
        polars="""result = (
    df.group_by("category")
      .agg(pl.col("id_column").n_unique().alias("distinct_ids"))
      .sort("distinct_ids", descending=True)
)""",
        pandas="""result = (
    df.groupby("category", as_index=False)["id_column"]
      .nunique()
      .rename(columns={"id_column": "distinct_ids"})
      .sort_values("distinct_ids", ascending=False)
)""",
    ),
    Recipe(
        title="Filter, then aggregate",
        asks=(
            "average order value for customers in EMEA",
            "total revenue where the channel is online",
            "how many rows match a condition",
        ),
        polars="""result = (
    df.filter(pl.col("category") == "the value")
      .select(
          pl.len().alias("n"),
          pl.col("measure").sum().alias("total_measure"),
      )
)""",
        pandas="""subset = df[df["category"] == "the value"]
result = pd.DataFrame({"n": [len(subset)], "total_measure": [subset["measure"].sum()]})""",
        trap=(
            "Report how many rows survived the filter. A total over three rows "
            "and a total over three thousand look identical otherwise."
        ),
    ),
    Recipe(
        title="Share of a total by category",
        asks=(
            "what percentage of revenue comes from each region",
            "share of total by product",
            "which category accounts for most of the total",
        ),
        polars="""result = (
    df.group_by("category")
      .agg(pl.col("measure").sum().alias("total_measure"))
      .with_columns(
          (pl.col("total_measure") / pl.col("total_measure").sum() * 100)
          .round(1).alias("pct_of_total")
      )
      .sort("pct_of_total", descending=True)
)""",
        pandas="""result = (
    df.groupby("category", as_index=False)["measure"].sum()
      .rename(columns={"measure": "total_measure"})
      .assign(pct_of_total=lambda d: (d["total_measure"] / d["total_measure"].sum() * 100).round(1))
      .sort_values("pct_of_total", ascending=False)
)""",
    ),
    Recipe(
        title="Missing values by column",
        asks=(
            "how many nulls are there",
            "which columns have missing data",
            "data quality summary",
        ),
        polars="""result = (
    df.null_count()
      .transpose(include_header=True, header_name="column", column_names=["nulls"])
      .with_columns((pl.col("nulls") / df.height * 100).round(1).alias("pct"))
      .filter(pl.col("nulls") > 0)
      .sort("nulls", descending=True)
)""",
        pandas="""nulls = df.isna().sum()
result = (
    nulls[nulls > 0]
    .rename_axis("column").reset_index(name="nulls")
    .assign(pct=lambda d: (d["nulls"] / len(df) * 100).round(1))
    .sort_values("nulls", ascending=False)
)""",
        trap=(
            "Missingness is rarely random. If the rate differs sharply between "
            "groups, the rows that remain are a biased sample and any average "
            "over them is biased too."
        ),
    ),
    Recipe(
        title="Outliers in a numeric column",
        asks=(
            "find the outliers in amount",
            "which rows are unusually large",
            "detect anomalies in a measure",
        ),
        polars="""q1 = df.select(pl.col("measure").quantile(0.25)).item()
q3 = df.select(pl.col("measure").quantile(0.75)).item()
iqr = q3 - q1
result = df.filter(
    (pl.col("measure") < q1 - 1.5 * iqr) | (pl.col("measure") > q3 + 1.5 * iqr)
).sort("measure", descending=True)""",
        pandas="""q1, q3 = df["measure"].quantile([0.25, 0.75])
iqr = q3 - q1
result = df[
    (df["measure"] < q1 - 1.5 * iqr) | (df["measure"] > q3 + 1.5 * iqr)
].sort_values("measure", ascending=False)""",
        trap=(
            "The IQR fence collapses when the middle half of the column is a "
            "single value, and then everything looks like an outlier. Sanity-check "
            "the count against the number of rows."
        ),
    ),
    Recipe(
        title="Cross-tabulation of two categories",
        asks=(
            "revenue by region and channel",
            "breakdown across two dimensions",
            "pivot category against category",
        ),
        polars="""result = (
    df.group_by(["row_category", "column_category"])
      .agg(pl.col("measure").sum().alias("total_measure"))
      .pivot(values="total_measure", index="row_category", on="column_category")
)""",
        pandas="""result = (
    df.pivot_table(
        values="measure", index="row_category", columns="column_category",
        aggfunc="sum", fill_value=0,
    ).reset_index()
)""",
    ),
    Recipe(
        title="Period over period change",
        asks=(
            "year over year growth",
            "how did this quarter compare with last",
            "change since the previous period",
        ),
        polars="""result = (
    df.with_columns(pl.col("date_column").dt.truncate("1mo").alias("period"))
      .group_by("period")
      .agg(pl.col("measure").sum().alias("total_measure"))
      .sort("period")
      .with_columns(
          (pl.col("total_measure").diff() / pl.col("total_measure").shift(1) * 100)
          .round(1).alias("pct_change")
      )
)""",
        pandas="""periods = (
    df.assign(period=pd.to_datetime(df["date_column"]).dt.to_period("M").dt.to_timestamp())
      .groupby("period", as_index=False)["measure"].sum()
      .rename(columns={"measure": "total_measure"})
      .sort_values("period")
)
result = periods.assign(pct_change=lambda d: (d["total_measure"].pct_change() * 100).round(1))""",
        trap=(
            "A percentage change against a near-zero base is enormous and means "
            "nothing. Keep the totals in the result so the reader can see the base."
        ),
    ),
    # The eight below close gaps a harvest of 179 ideation questions found: shapes
    # the datasets ask for that the fifteen above answer only by accident.
    Recipe(
        title="Rate of a yes or no outcome by category",
        asks=(
            "churn rate by region",
            "what percentage of customers in each segment defaulted",
            "how does the readmission rate vary across departments",
            "likelihood of recommending the service by channel",
        ),
        polars="""result = (
    df.group_by("category")
      .agg(
          (pl.col("flag_column").cast(pl.Boolean).mean() * 100).round(1).alias("rate_pct"),
          pl.len().alias("n"),
      )
      .filter(pl.col("n") >= 30)
      .sort("rate_pct", descending=True)
)""",
        pandas="""grouped = df.groupby("category")["flag_column"]
result = (
    grouped.agg(rate_pct=lambda s: round(s.astype(bool).mean() * 100, 1), n="size")
           .reset_index()
)
result = result[result["n"] >= 30].sort_values("rate_pct", ascending=False)""",
        trap=(
            "A rate is a mean of a 0/1 column, not a count — counting the yes rows "
            "gives the volume, not the rate, and ranks big groups top every time. "
            "Carry `n`: a 100% rate over three rows is noise, which is why the small "
            "groups are filtered out rather than shown and explained away."
        ),
    ),
    Recipe(
        title="Top N groups by an aggregate",
        asks=(
            "which cities have the highest average price",
            "which states have the highest average sales",
            "which five departments have the longest waiting time",
            "which product types earn the most per order",
        ),
        polars="""result = (
    df.group_by("category")
      .agg(
          pl.col("measure").mean().round(2).alias("avg_measure"),
          pl.len().alias("n"),
      )
      .filter(pl.col("n") >= 30)
      .sort("avg_measure", descending=True)
      .head(10)
)""",
        pandas="""result = (
    df.groupby("category")
      .agg(avg_measure=("measure", "mean"), n=("measure", "size"))
      .reset_index()
)
result = (
    result[result["n"] >= 30]
    .assign(avg_measure=lambda d: d["avg_measure"].round(2))
    .sort_values("avg_measure", ascending=False)
    .head(10)
)""",
        trap=(
            "This is not `sort().head()` on the rows — that returns the ten largest "
            "*rows*, and the question asked which groups rank highest on an average. "
            "Aggregate first, then sort. Without the size filter the top of the list "
            "is whichever group happens to have one member."
        ),
    ),
    Recipe(
        title="A measure by a numeric column cut into bands",
        asks=(
            "average satisfaction by age group",
            "how does satisfaction vary by household income",
            "default rate by tenure band",
            "does the effect change across income brackets",
        ),
        polars="""edges = [25, 40, 60]
result = (
    df.with_columns(
        pl.col("numeric_column")
          .cut(edges, labels=["<25", "25-39", "40-59", "60+"])
          .alias("band")
    )
    .group_by("band")
    .agg(pl.col("measure").mean().round(2).alias("avg_measure"), pl.len().alias("n"))
    .sort("band")
)""",
        pandas="""edges = [-float("inf"), 25, 40, 60, float("inf")]
bands = pd.cut(df["numeric_column"], bins=edges, labels=["<25", "25-39", "40-59", "60+"])
result = (
    df.assign(band=bands)
      .groupby("band", as_index=False, observed=True)
      .agg(avg_measure=("measure", "mean"), n=("measure", "size"))
)
result["avg_measure"] = result["avg_measure"].round(2)""",
        trap=(
            "Grouping on the raw numeric column makes one group per distinct value — "
            "hundreds of groups of one row. Bands are a choice, not a fact: state the "
            "edges in the answer, because moving them can move the conclusion. Sort by "
            "the band, never by the value, or the reader cannot read the trend."
        ),
    ),
    Recipe(
        title="Distribution of a numeric column",
        asks=(
            "what is the distribution of annual salaries",
            "distribution of patient satisfaction scores",
            "how are prices spread out",
            "what does the spread of subscriber counts look like",
        ),
        polars="""result = df.select(
    pl.col("numeric_column").count().alias("n"),
    pl.col("numeric_column").mean().round(2).alias("mean"),
    pl.col("numeric_column").median().round(2).alias("median"),
    pl.col("numeric_column").std().round(2).alias("std"),
    pl.col("numeric_column").min().alias("min"),
    pl.col("numeric_column").quantile(0.25).round(2).alias("p25"),
    pl.col("numeric_column").quantile(0.75).round(2).alias("p75"),
    pl.col("numeric_column").max().alias("max"),
)""",
        pandas="""stats = df["numeric_column"].describe(percentiles=[0.25, 0.5, 0.75])
result = stats.round(2).to_frame().T.reset_index(drop=True)""",
        trap=(
            "The distribution of a *numeric* column is quantiles, not `value_counts` — "
            "counting distinct salaries yields one row per employee. Report the median "
            "beside the mean: where they part company the column is skewed and the "
            "mean alone will mislead."
        ),
    ),
    Recipe(
        title="A measure by two categories at once",
        asks=(
            "how do impressions vary by channel and quarter",
            "average lift by segment for each metric",
            "sales by region and product type",
        ),
        polars="""result = (
    df.group_by(["first_category", "second_category"])
      .agg(pl.col("measure").mean().round(2).alias("avg_measure"), pl.len().alias("n"))
      .sort(["first_category", "second_category"])
)""",
        pandas="""result = (
    df.groupby(["first_category", "second_category"], as_index=False, observed=True)
      .agg(avg_measure=("measure", "mean"), n=("measure", "size"))
      .sort_values(["first_category", "second_category"])
)
result["avg_measure"] = result["avg_measure"].round(2)""",
        trap=(
            "Two keys multiply: ten channels by twelve quarters is 120 rows, many of "
            "them thin. Keep the result long rather than pivoting it wide — a long "
            "frame charts and reads back correctly, where a wide one hides the group "
            "sizes. Averaging the per-cell averages afterwards is not the overall "
            "average unless every cell holds the same number of rows."
        ),
    ),
    Recipe(
        title="Most common value per category",
        asks=(
            "most common waiting time across departments",
            "which product each region buys most",
            "the typical sleep disorder per occupation",
        ),
        polars="""result = (
    df.group_by(["category", "value_column"])
      .agg(pl.len().alias("n"))
      .sort(["category", "n"], descending=[False, True])
      .group_by("category", maintain_order=True)
      .first()
)""",
        pandas="""counts = (
    df.groupby(["category", "value_column"], observed=True)
      .size().reset_index(name="n")
      .sort_values(["category", "n"], ascending=[True, False])
)
result = counts.groupby("category", as_index=False, observed=True).first()""",
        trap=(
            "The mode is not the average, and `.mode()` returns every tied value, so a "
            "flat distribution yields several rows per group. Keep `n` in the result: "
            "a mode of 4 out of 400 is not what a reader hears in 'most common'."
        ),
    ),
    Recipe(
        title="Does one column affect another",
        asks=(
            "how does physical activity level affect stress",
            "how does age affect the likelihood of readmission",
            "does contract type influence churn",
            "what is the impact of occupation on sleep quality",
            "how does the use of AI affect audit effectiveness",
        ),
        polars="""# One question, two shapes — pick by the dtype of the column doing the affecting.
# Categorical driver: compare the outcome across its values.
result = (
    df.group_by("driver_column")
      .agg(pl.col("outcome_column").mean().round(2).alias("avg_outcome"), pl.len().alias("n"))
      .sort("avg_outcome", descending=True)
)
# Numeric driver, instead: correlate, or band it first with recipe 18.
# result = df.select(pl.corr("driver_column", "outcome_column").round(3).alias("pearson"))""",
        pandas="""# One question, two shapes — pick by the dtype of the column doing the affecting.
# Categorical driver: compare the outcome across its values.
result = (
    df.groupby("driver_column", as_index=False, observed=True)
      .agg(avg_outcome=("outcome_column", "mean"), n=("outcome_column", "size"))
      .sort_values("avg_outcome", ascending=False)
)
result["avg_outcome"] = result["avg_outcome"].round(2)
# Numeric driver, instead: correlate, or band it first with recipe 18.
# result = pd.DataFrame({"pearson": [df["driver_column"].corr(df["outcome_column"]).round(3)]})""",
        trap=(
            "'Affects' is a causal word and none of this establishes cause. Answer "
            "with the difference and let the reader draw the arrow: say group A "
            "averages 12 against group B's 9, never that A *causes* the 3. A "
            "difference across groups can also reverse once a third column is held "
            "fixed, so a large gap is a finding to check, not a conclusion."
        ),
    ),
    Recipe(
        title="Whether a difference between two groups is real",
        asks=(
            "which segments show statistically significant improvements",
            "is the difference between the variant and control real",
            "is that gap significant",
        ),
        polars="""summary = (
    df.group_by("group_column")
      .agg(
          pl.col("measure").mean().alias("mean"),
          pl.col("measure").std().alias("sd"),
          pl.len().alias("n"),
      )
)
rows = summary.to_dicts()
a, b = rows[0], rows[1]
se = (a["sd"] ** 2 / a["n"] + b["sd"] ** 2 / b["n"]) ** 0.5
diff = a["mean"] - b["mean"]
result = summary.with_columns(
    pl.lit(round(diff, 3)).alias("difference"),
    pl.lit(round(diff - 1.96 * se, 3)).alias("ci_low"),
    pl.lit(round(diff + 1.96 * se, 3)).alias("ci_high"),
)""",
        pandas="""summary = (
    df.groupby("group_column", observed=True)["measure"]
      .agg(mean="mean", sd="std", n="size").reset_index()
)
a, b = summary.iloc[0], summary.iloc[1]
se = (a["sd"] ** 2 / a["n"] + b["sd"] ** 2 / b["n"]) ** 0.5
diff = a["mean"] - b["mean"]
result = summary.assign(
    difference=round(diff, 3),
    ci_low=round(diff - 1.96 * se, 3),
    ci_high=round(diff + 1.96 * se, 3),
)""",
        trap=(
            "Report the interval, not a verdict. An interval straddling zero means the "
            "data does not settle the question — it is not evidence the groups are the "
            "same. Testing every segment until one looks significant is how noise gets "
            "published: with twenty segments, one crossing the line at the usual "
            "threshold is the expected result when nothing is going on. Say how many "
            "were examined, and keep the group sizes in the result."
        ),
    ),
)


def render(engine: str) -> str:
    """One recipe book, in one API's dialect."""
    lines = [
        f"# {engine.capitalize()} recipes",
        "",
        "Worked answers to the shapes a data question comes in. Find the shape that "
        "matches, then adapt the column names. Placeholder names like `category`, "
        "`measure` and `date_column` are stand-ins for real columns from the card.",
        "",
    ]
    for index, recipe in enumerate(RECIPES, 1):
        code = recipe.polars if engine == "polars" else recipe.pandas
        lines += [f"## {index}. {recipe.title}", ""]
        lines += ["Questions like: " + "; ".join(recipe.asks) + ".", ""]
        lines += ["```python", code, "```", ""]
        if recipe.trap:
            lines += [f"**Watch out.** {recipe.trap}", ""]
        for note in recipe.notes:
            lines += [note, ""]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for engine, name in (("polars", "recipes_polars.md"), ("pandas", "recipes_pandas.md")):
        text = render(engine)
        (OUT / name).write_text(text, encoding="utf-8")
        print(f"wrote {name}: {len(RECIPES)} recipes, {len(text):,} bytes")


if __name__ == "__main__":
    main()
