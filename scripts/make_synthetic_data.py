"""Generate a synthetic corpus for exercising insightsmith end to end.

Real files are the only honest test, but the ones to hand are a handful and none
of them has a *known* answer. These are built with the relationship planted on
purpose, so a run can be judged against what is actually in the data rather than
against whether the output reads plausibly.

Every format the loader claims is represented, and every dataset carries at
least one thing a consultant meets and a naive analysis gets wrong: diminishing
returns read as linear, a paradox that reverses under segmentation, sixty
p-values with three significant by chance, numbers stored as text, missingness
that is not random.

Seeded, so a regression is a change in insightsmith rather than in the data::

    python scripts/make_synthetic_data.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data_example" / "synthetic"
SEED: Final = 20260909

#: What each file plants, written out beside the data so a result can be judged.
TRUTHS: dict[str, list[str]] = {}


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


# --------------------------------------------------------------------------- #
# 1. csv — trend, seasonality, and a correlation that is genuinely there
# --------------------------------------------------------------------------- #


def retail_sales() -> pl.DataFrame:
    rng = _rng()
    n = 5_000
    day = rng.integers(0, 730, n)
    date = np.datetime64("2024-01-01") + day.astype("timedelta64[D]")
    month = date.astype("datetime64[M]").astype(int) % 12 + 1

    region = rng.choice(["North", "South", "East", "West"], n, p=[0.3, 0.3, 0.25, 0.15])
    category = rng.choice(["Coffee", "Tea", "Bakery", "Cold Drinks"], n, p=[0.4, 0.25, 0.2, 0.15])
    channel = rng.choice(["Store", "Online", "Wholesale"], n, p=[0.55, 0.35, 0.10])

    units = rng.poisson(12, n) + 1
    price = np.round(rng.normal(4.2, 0.8, n).clip(1.5, 9.0), 2)
    # December lifts volume; West declines steadily while the others grow.
    seasonal = np.where(month == 12, 1.42, 1.0)
    drift = 1.0 + day / 730 * np.where(region == "West", -0.28, 0.22)
    revenue = np.round(units * price * seasonal * drift, 2)
    cost = np.round(revenue * rng.uniform(0.52, 0.71, n), 2)

    return pl.DataFrame(
        {
            "order_date": [str(d) for d in date],
            "region": region,
            "product_category": category,
            "channel": channel,
            "units": units,
            "unit_price": price,
            "revenue": revenue,
            "cost": cost,
        }
    ).sort("order_date")


TRUTHS["retail_sales.csv"] = [
    "revenue correlates with units, but price varies so it is not near-perfect",
    "December revenue runs about 42% above other months",
    "West is the only region trending down; the other three grow across 2024-2025",
    "margin = (revenue - cost) / revenue sits near 39%",
]


# --------------------------------------------------------------------------- #
# 2. tsv — diminishing returns, which Pearson understates
# --------------------------------------------------------------------------- #


def marketing_spend() -> pl.DataFrame:
    rng = _rng()
    n = 400
    spend = np.round(np.exp(rng.normal(8.2, 0.9, n)).clip(200, 250_000), 2)
    # Conversions saturate: doubling spend does not double the result.
    conversions = np.round(38 * np.log(spend) - 180 + rng.normal(0, 22, n)).clip(0).astype(int)
    return pl.DataFrame(
        {
            "campaign_id": [f"CMP-{i:04d}" for i in range(n)],
            "channel": rng.choice(["Search", "Social", "Display", "Email"], n),
            "quarter": rng.choice(["2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4"], n),
            "spend_usd": spend,
            "impressions": (spend * rng.uniform(18, 42, n)).astype(int),
            "conversions": conversions,
        }
    )


TRUTHS["marketing_spend.tsv"] = [
    "conversions follow log(spend), not spend, so returns diminish sharply",
    "Pearson on the raw columns understates it; Spearman is much higher",
    "spend is log-normal, so its mean sits far above its median",
]


# --------------------------------------------------------------------------- #
# 3. parquet — an imbalanced target, one real driver and one red herring
# --------------------------------------------------------------------------- #


def customer_churn() -> pl.DataFrame:
    rng = _rng()
    n = 8_000
    tenure = rng.integers(1, 72, n)
    tickets = rng.poisson(1.6, n)
    monthly = np.round(rng.normal(64, 21, n).clip(15, 180), 2)
    region = rng.choice(["EMEA", "AMER", "APAC"], n)  # deliberately unrelated

    # Churn falls with tenure and rises with unresolved contact.
    logit = 1.1 - 0.055 * tenure + 0.38 * tickets + 0.004 * (monthly - 64)
    churned = rng.random(n) < 1 / (1 + np.exp(-logit)) * 0.42

    return pl.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "tenure_months": tenure,
            "support_tickets": tickets,
            "monthly_charge": monthly,
            "region": region,
            "contract": rng.choice(["Monthly", "Annual", "Two year"], n, p=[0.55, 0.3, 0.15]),
            "churned": churned,
        }
    )


TRUTHS["customer_churn.parquet"] = [
    "the target is imbalanced, so accuracy is a useless measure here",
    "tenure_months is the strongest driver: churn falls sharply as tenure rises",
    "support_tickets raises churn; monthly_charge matters only slightly",
    "region is pure noise and any difference between EMEA/AMER/APAC is chance",
]


# --------------------------------------------------------------------------- #
# 4. json — records keyed by id, with every value stored as text
# --------------------------------------------------------------------------- #


def clinic_visits() -> dict[str, dict[str, str]]:
    rng = _rng()
    departments = ["Cardiology", "Oncology", "Paediatrics", "Neurology", "Orthopaedics"]
    out: dict[str, dict[str, str]] = {}
    for i in range(1_200):
        wait = int(rng.gamma(3.0, 9.0))
        out[f"VISIT-{i:05d}"] = {
            "department": str(rng.choice(departments)),
            "wait_minutes": str(wait),
            "satisfaction": str(
                round(float(np.clip(5.6 - wait * 0.035 + rng.normal(0, 0.9), 1, 10)), 1)
            ),
            "age": str(int(rng.integers(1, 96))),
            "readmitted_30d": "yes" if rng.random() < 0.11 else "no",
            "cost_eur": f"{rng.gamma(4.0, 320.0):.2f}",
        }
    return out


TRUTHS["clinic_visits.json"] = [
    "the document is keyed by visit id, so it must be read as 1200 rows not 1 wide row",
    "every value is a string; arithmetic needs a cast first",
    "satisfaction falls as wait_minutes rises (negative correlation around -0.5)",
    "wait_minutes and cost_eur are gamma-shaped, so both are right-skewed",
]


# --------------------------------------------------------------------------- #
# 5. jsonl — sixty subgroup tests, three significant purely by chance
# --------------------------------------------------------------------------- #


def ab_test_subgroups() -> list[dict[str, object]]:
    rng = _rng()
    segments = [
        f"{a}-{b}"
        for a in ("EMEA", "AMER", "APAC", "LATAM")
        for b in ("18-24", "25-34", "35-44", "45-54", "55+")
    ]
    rows: list[dict[str, object]] = []
    for segment in segments:
        for metric in ("signup_rate", "retention_7d", "revenue_per_user"):
            # No true effect anywhere: p is uniform, so about 3 of 60 land under 0.05.
            rows.append(
                {
                    "segment": segment,
                    "metric": metric,
                    "n_control": int(rng.integers(400, 4_000)),
                    "n_variant": int(rng.integers(400, 4_000)),
                    "lift_pct": round(float(rng.normal(0, 2.4)), 3),
                    "p_value": round(float(rng.random()), 4),
                }
            )
    return rows


TRUTHS["ab_test_subgroups.jsonl"] = [
    "there is NO real effect anywhere: every p_value is drawn uniformly",
    "about 3 of the 60 tests fall below 0.05, which is exactly what chance predicts",
    "any 'winning segment' read off this without correction is a false positive",
]


# --------------------------------------------------------------------------- #
# 6. xlsx — Simpson's paradox, plus PII and a heavy salary tail
# --------------------------------------------------------------------------- #


def hr_compensation() -> pl.DataFrame:
    rng = _rng()
    levels = ["Associate", "Senior", "Lead", "Principal", "Director"]
    base = {
        "Associate": 52_000,
        "Senior": 78_000,
        "Lead": 104_000,
        "Principal": 148_000,
        "Director": 210_000,
    }
    rows = []
    for i in range(900):
        # Women are concentrated in junior levels, and paid slightly more within
        # each level. The aggregate gap therefore points the other way.
        gender = "F" if rng.random() < 0.46 else "M"
        weights = (
            [0.44, 0.30, 0.15, 0.08, 0.03] if gender == "F" else [0.24, 0.26, 0.22, 0.18, 0.10]
        )
        level = str(rng.choice(levels, p=weights))
        salary = base[level] * (1.012 if gender == "F" else 1.0) * rng.lognormal(0, 0.11)
        rows.append(
            {
                "employee_id": f"E{i:05d}",
                "full_name": f"Employee {i:05d}",
                "email": f"employee{i:05d}@example.com",
                "phone": f"+1 555 {rng.integers(1000, 9999)}",
                "department": str(rng.choice(["Engineering", "Sales", "Finance", "People"])),
                "job_level": level,
                "gender": gender,
                "years_experience": int(np.clip(rng.normal(levels.index(level) * 4 + 3, 2), 0, 40)),
                "annual_salary": round(float(salary), 2),
            }
        )
    return pl.DataFrame(rows)


TRUTHS["hr_compensation.xlsx"] = [
    "SIMPSON'S PARADOX: overall, women's mean salary is LOWER than men's",
    "within every job_level, women are paid MORE",
    "the aggregate gap is composition: women are concentrated at junior levels",
    "full_name, email and phone are PII and must be masked before any model sees them",
    "annual_salary is log-normal, so the mean overstates a typical salary",
]


# --------------------------------------------------------------------------- #
# 7. csv — the European shape: semicolons, decimal commas, cp1252, DD.MM.YYYY
# --------------------------------------------------------------------------- #


def umsatz_regional() -> str:
    """Semicolons, decimal commas, cp1252 umlauts and DD.MM.YYYY dates.

    The shape a European export actually arrives in, and the one where guessing
    the delimiter or the codepage wrong is silent rather than loud.
    """
    rng = _rng()
    regions = ["Nord", "S\u00fcd", "Ost", "West", "Z\u00fcrich", "M\u00fcnchen"]
    categories = ["Getr\u00e4nke", "S\u00fc\u00dfwaren", "Backwaren"]
    lines = ["Datum;Region;Menge;Umsatz;Kategorie"]
    for _ in range(600):
        day = 1 + int(rng.integers(0, 28))
        month = 1 + int(rng.integers(0, 12))
        menge = int(rng.integers(1, 90))
        umsatz = f"{menge * float(rng.normal(11.4, 2.2)):.2f}".replace(".", ",")
        lines.append(
            f"{day:02d}.{month:02d}.2025;{rng.choice(regions)};{menge};"
            f"{umsatz};{rng.choice(categories)}"
        )
    return "\n".join(lines) + "\n"


TRUTHS["umsatz_regional.csv"] = [
    "semicolon-delimited, decimal comma, cp1252, dates as DD.MM.YYYY",
    "region names carry umlauts (Süd, Zürich, München) which is what breaks a wrong codepage",
    "Umsatz is roughly Menge * 11.4, so the two correlate strongly once parsed as numbers",
]


# --------------------------------------------------------------------------- #
# 8. arrow — seasonality, injected anomalies, and sensors with one reading
# --------------------------------------------------------------------------- #


def sensor_readings() -> pl.DataFrame:
    rng = _rng()
    n = 20_000
    hour = rng.integers(0, 24 * 90, n)
    sensor = rng.choice([f"S-{i:02d}" for i in range(18)], n)
    # Daily cycle plus a slow drift, then 0.4% of readings pushed far out.
    temp = 21 + 4.5 * np.sin(hour / 24 * 2 * math.pi) + hour / (24 * 90) * 2.2
    temp = temp + rng.normal(0, 0.8, n)
    spikes = rng.random(n) < 0.004
    temp[spikes] += rng.choice([-1, 1], spikes.sum()) * rng.uniform(14, 30, spikes.sum())

    frame = pl.DataFrame(
        {
            "reading_hour": hour,
            "sensor_id": sensor,
            "temperature_c": np.round(temp, 3),
            "humidity_pct": np.round(rng.normal(48, 9, n).clip(5, 99), 1),
            "site": rng.choice(["Plant A", "Plant B", "Plant C"], n, p=[0.5, 0.35, 0.15]),
        }
    ).sort("reading_hour")
    # Three sensors that reported once, so any per-sensor average rests on one row.
    lonely = pl.DataFrame(
        {
            "reading_hour": [2160, 2161, 2162],
            "sensor_id": ["S-98", "S-99", "S-97"],
            "temperature_c": [23.5, 19.2, 27.1],
            "humidity_pct": [44.0, 51.0, 39.5],
            "site": ["Plant C", "Plant C", "Plant B"],
        },
        schema=frame.schema,
    )
    return pl.concat([frame, lonely])


TRUTHS["sensor_readings.arrow"] = [
    "temperature follows a 24-hour cycle plus a slow upward drift of about 2C over 90 days",
    "roughly 0.4% of readings are injected anomalies 14-30C away from the curve",
    "sensors S-97, S-98 and S-99 have exactly ONE reading each",
    "a per-sensor average therefore includes three groups of size 1",
]


# --------------------------------------------------------------------------- #
# 9. csv — missingness that is not random
# --------------------------------------------------------------------------- #


def survey_responses() -> pl.DataFrame:
    rng = _rng()
    n = 2_400
    satisfaction = rng.integers(1, 6, n)
    income = rng.lognormal(10.8, 0.55, n)
    # Dissatisfied respondents skip the income question far more often, so the
    # observed income distribution is biased upward.
    drop = rng.random(n) < np.where(satisfaction <= 2, 0.55, 0.06)
    income_text = [None if d else f"{v:,.0f}" for d, v in zip(drop, income, strict=True)]

    return pl.DataFrame(
        {
            "respondent_id": [f"R{i:05d}" for i in range(n)],
            "satisfaction": satisfaction,
            "would_recommend": rng.choice(["Yes", "No", "Not sure"], n, p=[0.55, 0.2, 0.25]),
            "tenure_band": rng.choice(["<1y", "1-3y", "3-5y", "5y+"], n),
            "household_income": income_text,
            "channel": rng.choice(["Email", "In-app", "Phone"], n, p=[0.5, 0.4, 0.1]),
        }
    )


TRUTHS["survey_responses.csv"] = [
    "household_income is MISSING NOT AT RANDOM: dissatisfied respondents skip it 55% of the time",
    "so mean income among respondents overstates the population",
    "household_income is written with thousands separators, so it loads as text",
    "about 15% of household_income values are null overall",
]


# --------------------------------------------------------------------------- #


def _corr_text(frame: pl.DataFrame, left: str, right: str) -> float:
    """Correlation between two columns that arrived as text."""
    cast = frame.select(
        pl.col(left).cast(pl.Float64, strict=False),
        pl.col(right).cast(pl.Float64, strict=False),
    )
    return float(cast.select(pl.corr(left, right)).item())


def _measure() -> dict[str, list[str]]:
    """Read the files back and state what is actually in them.

    Prose drifts from data. Three of these claims were wrong when written by
    hand, which makes a corpus worse than no corpus: a run gets judged against a
    number nobody checked. So the headline figures are computed from the files
    after they are written, and the README carries what was measured.
    """
    from insightsmith.io.loaders import load
    from insightsmith.io.sniff import sniff

    def frame(name: str) -> pl.DataFrame:
        return load(sniff(OUT / name)).collect()

    out: dict[str, list[str]] = {}

    retail = frame("retail_sales.csv")
    margin = (retail["revenue"].sum() - retail["cost"].sum()) / retail["revenue"].sum()
    out["retail_sales.csv"] = [
        f"corr(units, revenue) = {retail.select(pl.corr('units', 'revenue')).item():.3f}",
        f"overall margin = {margin:.1%}",
        f"{retail.height:,} rows",
    ]

    spend = frame("marketing_spend.tsv")
    pearson = spend.select(pl.corr("spend_usd", "conversions")).item()
    spearman = spend.select(pl.corr("spend_usd", "conversions", method="spearman")).item()
    out["marketing_spend.tsv"] = [
        f"Pearson = {pearson:.3f} but Spearman = {spearman:.3f}, the gap curvature leaves",
        f"spend mean = {spend['spend_usd'].mean():,.0f} against median "
        f"{spend['spend_usd'].median():,.0f}",
    ]

    churn = frame("customer_churn.parquet")
    by_region = churn.group_by("region").agg(pl.col("churned").mean()).sort("region")
    rates = ", ".join(f"{r['region']} {r['churned']:.3f}" for r in by_region.to_dicts())
    out["customer_churn.parquet"] = [
        f"overall churn rate = {churn['churned'].mean():.1%}",
        f"by region (noise by construction): {rates}",
    ]

    hr = frame("hr_compensation.xlsx")
    agg = hr.group_by("gender").agg(pl.col("annual_salary").mean()).sort("gender")
    women, men = (row["annual_salary"] for row in agg.to_dicts())
    within = (
        hr.group_by(["job_level", "gender"])
        .agg(pl.col("annual_salary").mean())
        .pivot(values="annual_salary", index="job_level", on="gender")
        .with_columns(((pl.col("F") / pl.col("M") - 1) * 100).alias("gap"))
        .sort("gap")
    )
    gaps = ", ".join(f"{r['job_level']} {r['gap']:+.1f}%" for r in within.to_dicts())
    out["hr_compensation.xlsx"] = [
        f"aggregate: women {women:,.0f} against men {men:,.0f}, a gap of {men - women:,.0f}",
        f"within each level, women earn MORE: {gaps}",
        "so the aggregate reverses the within-level direction, which is the paradox",
    ]

    ab = frame("ab_test_subgroups.jsonl")
    significant = ab.filter(pl.col("p_value") < 0.05).height
    out["ab_test_subgroups.jsonl"] = [
        f"{significant} of {ab.height} tests fall below p < 0.05, against "
        f"{ab.height * 0.05:.1f} expected by chance alone",
    ]

    survey = frame("survey_responses.csv")
    missing = (
        survey.group_by("satisfaction")
        .agg(pl.col("household_income").is_null().mean())
        .sort("satisfaction")
    )
    rates = ", ".join(
        f"{r['satisfaction']}: {r['household_income']:.0%}" for r in missing.to_dicts()
    )
    out["survey_responses.csv"] = [
        f"null rate overall = {survey['household_income'].is_null().mean():.1%}",
        f"null rate by satisfaction score, which is what makes it MNAR: {rates}",
    ]

    sensors = frame("sensor_readings.arrow")
    counts = sensors.group_by("sensor_id").len().filter(pl.col("len") < 5).sort("sensor_id")
    out["sensor_readings.arrow"] = [
        f"{sensors.height:,} rows across {sensors['sensor_id'].n_unique()} sensors",
        "sensors with fewer than 5 readings: "
        + ", ".join(f"{r['sensor_id']} ({r['len']})" for r in counts.to_dicts()),
    ]

    clinic = frame("clinic_visits.json")
    out["clinic_visits.json"] = [
        f"read as {clinic.height:,} rows x {clinic.width} columns, not one wide row",
        "corr(wait_minutes, satisfaction) = "
        f"{_corr_text(clinic, 'wait_minutes', 'satisfaction'):.3f}",
    ]

    umsatz = frame("umsatz_regional.csv")
    out["umsatz_regional.csv"] = [
        f"corr(Menge, Umsatz) = {umsatz.select(pl.corr('Menge', 'Umsatz')).item():.3f}",
        "regions read back with umlauts intact: "
        + ", ".join(sorted(umsatz["Region"].unique().to_list())),
    ]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    retail_sales().write_csv(OUT / "retail_sales.csv")
    marketing_spend().write_csv(OUT / "marketing_spend.tsv", separator="\t")
    customer_churn().write_parquet(OUT / "customer_churn.parquet")
    (OUT / "clinic_visits.json").write_text(json.dumps(clinic_visits()), encoding="utf-8")
    (OUT / "ab_test_subgroups.jsonl").write_text(
        "\n".join(json.dumps(r) for r in ab_test_subgroups()) + "\n", encoding="utf-8"
    )
    hr_compensation().write_excel(OUT / "hr_compensation.xlsx")
    (OUT / "umsatz_regional.csv").write_bytes(umsatz_regional().encode("cp1252"))
    sensor_readings().write_ipc(OUT / "sensor_readings.arrow")
    survey_responses().write_csv(OUT / "survey_responses.csv")

    readme = [
        "# Synthetic corpus",
        "",
        "Generated by `scripts/make_synthetic_data.py`, seeded so results are comparable",
        "between runs. Each file plants a relationship on purpose, listed below, so a run",
        "can be judged against what is in the data rather than against whether it reads",
        "plausibly.",
        "",
    ]
    measured = _measure()
    for name, truths in TRUTHS.items():
        size = (OUT / name).stat().st_size
        readme += [f"## {name}  ({size / 1024:,.0f} KB)", "", "**Planted**", ""]
        readme += [f"- {truth}" for truth in truths]
        if name in measured:
            readme += ["", "**Measured on the file as written**", ""]
            readme += [f"- {line}" for line in measured[name]]
        readme.append("")
    (OUT / "README.md").write_text("\n".join(readme), encoding="utf-8")

    for path in sorted(OUT.iterdir()):
        print(f"  {path.name:28} {path.stat().st_size / 1024:>9,.0f} KB")


if __name__ == "__main__":
    main()
