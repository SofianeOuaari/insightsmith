"""Best-effort masking of obvious personal data before anything reaches a model.

**This reduces exposure; it does not guarantee privacy.** It catches values that
look like contact details or identifiers, and blanks columns whose *names* say
they hold personal data. It cannot recognise a person's name in free text, an
address split across columns, or an identifier in a format it has never seen. If
data must not leave the machine, set ``local_only`` and keep it local — masking
is defence in depth, not a substitute for it.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "SENSITIVE_NAME",
    "is_sensitive_column",
    "mask_text",
    "mask_value",
]

#: Column names that hold personal data whatever the values happen to look like.
SENSITIVE_NAME: Final = re.compile(
    r"(^|_|\b)("
    r"name|firstname|first_name|lastname|last_name|surname|fullname|full_name|"
    r"email|e_mail|mail|phone|telephone|tel|mobile|msisdn|fax|"
    r"address|street|addr|postcode|postal_code|zip|zipcode|"
    r"ssn|sin|nino|national_id|passport|licence|license|"
    r"iban|bic|swift|account_number|card_number|credit_card|cc_number|"
    r"dob|date_of_birth|birth_date|birthdate|"
    r"latitude|longitude|lat|lon|lng|ip|ip_address|user_agent"
    r")($|_|\b)",
    re.I,
)

_EMAIL: Final = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]{2,}")
_IPV4: Final = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_IBAN: Final = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
_NATIONAL_ID: Final = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Deliberately loose; the Luhn check below decides whether it is really a card.
_CARD_CANDIDATE: Final = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PHONE: Final = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,17}\d(?!\w)")

_MASK: Final = {
    "email": "<email>",
    "ip": "<ip>",
    "iban": "<iban>",
    "national_id": "<id>",
    "card": "<card>",
    "phone": "<phone>",
}


def is_sensitive_column(name: str) -> bool:
    """Whether a column's *name* declares it personal.

    Name-based masking catches what value patterns cannot — a ``surname`` column
    holds ordinary words, and no regex will tell them from any other text.
    """
    return bool(SENSITIVE_NAME.search(name))


def mask_text(text: str) -> tuple[str, set[str]]:
    """Replace recognisable personal values, returning the text and what was found."""
    found: set[str] = set()

    def swap(pattern: re.Pattern[str], kind: str, value: str) -> str:
        def replace(match: re.Match[str]) -> str:
            found.add(kind)
            return _MASK[kind]

        return pattern.sub(replace, value)

    masked = swap(_EMAIL, "email", text)
    masked = swap(_IBAN, "iban", masked)
    masked = swap(_NATIONAL_ID, "national_id", masked)
    masked = swap(_IPV4, "ip", masked)

    # Card numbers before phones: both match long digit runs, and a Luhn-valid
    # number is far more likely to be a card than a phone.
    masked = _CARD_CANDIDATE.sub(lambda m: _card_or_keep(m.group(0), found), masked)
    return swap(_PHONE, "phone", masked), found


def mask_value(value: object, *, column: str = "") -> tuple[str, set[str]]:
    """Mask a single cell, taking the column name into account.

    A sensitive column name blanks the value outright; the model needs to know
    the column exists and what shape it has, never what is in it.
    """
    if value is None:
        return "", set()
    text = str(value)
    if column and is_sensitive_column(column):
        return "<redacted>", {"column_name"}
    return mask_text(text)


def _card_or_keep(candidate: str, found: set[str]) -> str:
    digits = re.sub(r"\D", "", candidate)
    if 13 <= len(digits) <= 19 and _luhn(digits):
        found.add("card")
        return _MASK["card"]
    return candidate


def _luhn(digits: str) -> bool:
    """Luhn checksum, so ordinary long numbers are not mistaken for cards."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0
