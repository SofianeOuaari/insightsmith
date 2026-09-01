"""Dataset card and PII masking.

The card is the only thing an agent ever sees, so these tests are about what it
must *not* contain as much as what it must.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from insightsmith.io.sniff import sniff
from insightsmith.profiling import profile_with_sample
from insightsmith.profiling.card import MAX_CARD_BYTES, DatasetCard, build_card
from insightsmith.profiling.pii import is_sensitive_column, mask_text, mask_value


def _card(path: Path, **kwargs):
    result, sample = profile_with_sample(sniff(path))
    return build_card(result, sample, **kwargs)


# --------------------------------------------------------------------------- #
# PII masking
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "secret", "kind"),
    [
        ("write to ada@example.com please", "ada@example.com", "email"),
        ("server at 192.168.1.1 responded", "192.168.1.1", "ip"),
        ("GB29NWBK60161331926819 is the account", "GB29NWBK60161331926819", "iban"),
        ("ssn 123-45-6789 on file", "123-45-6789", "national_id"),
        ("call +44 20 7946 0958 today", "7946", "phone"),
    ],
)
def test_recognisable_values_are_masked(raw: str, secret: str, kind: str) -> None:
    masked, kinds = mask_text(raw)
    assert kind in kinds
    assert secret not in masked, f"{secret} survived masking"


def test_card_numbers_are_luhn_checked() -> None:
    """A Luhn-valid number is a card; an arbitrary long number is not."""
    masked, kinds = mask_text("4111111111111111")
    assert "card" in kinds
    assert "4111" not in masked

    masked, kinds = mask_text("1234567890123456")
    assert "card" not in kinds


def test_sensitive_column_names_are_recognised() -> None:
    for name in ("email", "customer_name", "home_address", "phone_number", "dob", "iban"):
        assert is_sensitive_column(name), name
    for name in ("revenue", "region", "units", "signup_source"):
        assert not is_sensitive_column(name), name


def test_a_sensitive_column_blanks_the_value_whatever_it_holds() -> None:
    """A surname column holds ordinary words; no regex would catch them."""
    masked, kinds = mask_value("Okonkwo", column="last_name")
    assert masked == "<redacted>"
    assert "column_name" in kinds


def test_ordinary_values_survive_untouched() -> None:
    masked, kinds = mask_value("north", column="region")
    assert masked == "north"
    assert not kinds


def test_none_becomes_empty() -> None:
    assert mask_value(None) == ("", set())


# --------------------------------------------------------------------------- #
# card contents
# --------------------------------------------------------------------------- #


def test_card_describes_the_data(samples: dict[str, Path]) -> None:
    card = _card(samples["csv"])
    assert card.n_rows == 3
    assert card.column_names() == {"region", "units", "revenue"}
    assert card.name == "plain.csv"


def test_card_is_valid_json_and_hashes_stably(samples: dict[str, Path]) -> None:
    card = _card(samples["csv"])
    assert json.loads(card.to_json())
    assert card.hash == _card(samples["csv"]).hash  # same data, same plan


def test_different_data_gives_a_different_hash(samples: dict[str, Path]) -> None:
    assert _card(samples["csv"]).hash != _card(samples["messy"]).hash


def test_card_stays_within_budget_however_wide_the_data(tmp_path: Path) -> None:
    """Flat token cost is the entire point; a card that grows defeats it."""
    columns = 60
    header = ",".join(f"col_{i}" for i in range(columns))
    rows = "\n".join(",".join(str(i + j) for j in range(columns)) for i in range(500))
    path = tmp_path / "wide.csv"
    path.write_text(f"{header}\n{rows}\n", encoding="utf-8")

    card = _card(path)
    assert card.size_bytes <= MAX_CARD_BYTES
    # Every column must survive: an agent that cannot see one will invent it.
    assert len(card.columns) == columns


def test_card_never_contains_a_sensitive_column_value(tmp_path: Path) -> None:
    path = tmp_path / "people.csv"
    path.write_text(
        "customer_name,email,region,spend\n"
        "Ada Lovelace,ada@example.com,north,120\n"
        "Alan Turing,alan@example.com,south,80\n"
        "Grace Hopper,grace@example.com,east,95\n",
        encoding="utf-8",
    )
    card = _card(path)
    body = card.to_json()

    for leaked in ("Ada Lovelace", "ada@example.com", "Alan Turing", "Grace Hopper"):
        assert leaked not in body, f"{leaked} reached the card"
    assert "customer_name" in card.redacted_columns
    assert "email" in card.redacted_columns
    # The non-sensitive columns must still be described.
    assert "region" in card.column_names()


def test_examples_can_be_omitted_entirely(tmp_path: Path) -> None:
    path = tmp_path / "s.csv"
    path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
    assert _card(path, include_examples=False).examples == []
    assert _card(path).examples


def test_examples_span_the_data_rather_than_its_head(tmp_path: Path) -> None:
    """A sorted file makes the first rows unrepresentative."""
    path = tmp_path / "ordered.csv"
    rows = "\n".join(str(i) for i in range(200))
    path.write_text(f"n\n{rows}\n", encoding="utf-8")
    values = [int(row["n"]) for row in _card(path).examples]
    assert max(values) > 100


def test_correlations_are_reported_strongest_first(tmp_path: Path) -> None:
    path = tmp_path / "corr.csv"
    lines = "\n".join(f"{i},{i * 2},{(i * 7) % 13}" for i in range(60))
    path.write_text(f"x,y,noise\n{lines}\n", encoding="utf-8")
    card = _card(path)
    assert card.correlations
    assert {card.correlations[0]["a"], card.correlations[0]["b"]} == {"x", "y"}
    assert abs(float(card.correlations[0]["r"])) > 0.9


def test_quality_notes_reach_the_card(samples: dict[str, Path]) -> None:
    card = _card(samples["messy"])
    assert any(item["issue"] == "duplicate_rows" for item in card.quality)


def test_card_without_a_frame_still_describes_the_data(samples: dict[str, Path]) -> None:
    result, _ = profile_with_sample(sniff(samples["csv"]))
    card = build_card(result)
    assert card.columns
    assert card.examples == []
    assert card.correlations == []


def test_masked_kinds_are_listed_as_the_token_not_as_a_bare_word() -> None:
    """ "phone" in a card whose other lists are column names reads as a column.

    That is not hypothetical: a model short of usable columns wrote
    `pl.col("phone")` against a dataset of Pokemon.
    """
    card = DatasetCard(name="d", n_rows=1, n_columns=1, estimated=False)
    card.masked_kinds = ["phone", "email"]

    payload = card.to_dict()

    assert payload["masked"] == ["<phone>", "<email>"]
    assert "phone" not in json.dumps(payload["masked"]).replace("<phone>", "")


def test_a_very_wide_frame_says_how_much_it_could_not_show(tmp_path: Path) -> None:
    """Silently truncating the schema is what misleads; saying so is not."""
    names = [f"measurement_column_{i}" for i in range(600)]
    path = tmp_path / "wide.csv"
    path.write_text(
        ",".join(names) + "\n" + ",".join(str(i) for i in range(600)) + "\n", encoding="utf-8"
    )
    result, frame = profile_with_sample(sniff(path))

    card = build_card(result, frame)

    assert card.size_bytes <= MAX_CARD_BYTES
    assert card.omitted_columns > 0
    assert card.to_dict()["columns_omitted_for_size"] == card.omitted_columns
    assert len(card.columns) + card.omitted_columns == card.n_columns


def test_an_ordinary_frame_omits_nothing(tmp_path: Path) -> None:
    path = tmp_path / "small.csv"
    path.write_text("region,revenue\nnorth,1.0\nsouth,2.0\n", encoding="utf-8")
    result, frame = profile_with_sample(sniff(path))
    card = build_card(result, frame)

    assert card.omitted_columns == 0
    assert "columns_omitted_for_size" not in card.to_dict()


def test_numeric_prose_in_brackets_is_not_a_phone_number() -> None:
    """The chain this broke: a false positive lands "phone" on the card, and a
    model short of columns writes `pl.col("phone")` against a Pokemon dataset."""
    for text in (
        "200 (26.1% with PokeBall, full HP)",
        "20 (4,884-5,140 steps)",
        "50 (normal)",
        "135.5 kg (298.7 lbs)",
    ):
        masked, kinds = mask_text(text)
        assert kinds == set(), f"{text!r} was masked as {kinds}"
        assert masked == text


def test_real_phone_numbers_are_still_masked() -> None:
    for text in ("+1 (555) 123-4567", "555-123-4567", "call 07700 900123 now"):
        _, kinds = mask_text(text)
        assert kinds == {"phone"}, text
