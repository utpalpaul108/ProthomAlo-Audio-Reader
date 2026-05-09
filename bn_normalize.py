"""
Bangla text normalization for natural TTS reading.

edge_tts pronounces digits inconsistently (sometimes English, sometimes
digit-by-digit). For radio-newscast feel we expand all numbers to Bangla words
before synthesis. Also handles dates, currency, percentages, and phone numbers
context-sensitively.

Public entrypoint: normalize_all(text) -> str
Order of passes (most specific first): phone → date → year → currency →
percentage → general digits.

Implementation note: each pass uses re.sub with a callback so already-converted
text (Bangla words) is not re-matched by later passes.
"""

from __future__ import annotations

import re
from typing import Optional

from num2words import num2words

# ────────────────────────────────────────────────────────────────────────────
# Digit translation
# ────────────────────────────────────────────────────────────────────────────
_BN_DIGITS = "০১২৩৪৫৬৭৮৯"
_EN_DIGITS = "0123456789"
_BN_TO_EN = str.maketrans(_BN_DIGITS, _EN_DIGITS)
_EN_TO_BN_WORD = ["শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়"]

_BN_MONTHS = {
    1: "জানুয়ারি", 2: "ফেব্রুয়ারি", 3: "মার্চ", 4: "এপ্রিল",
    5: "মে", 6: "জুন", 7: "জুলাই", 8: "আগস্ট",
    9: "সেপ্টেম্বর", 10: "অক্টোবর", 11: "নভেম্বর", 12: "ডিসেম্বর",
}


def to_en_digits(s: str) -> str:
    """Translate Bangla digits to ASCII digits, leaving everything else alone."""
    return s.translate(_BN_TO_EN)


def to_bn_words(n: int) -> str:
    """Wraps num2words(n, lang='bn'). Defensive against the rare case where
    num2words emits the older Indian-style লক্ষ (BD prefers লাখ)."""
    out = num2words(n, lang="bn")
    return out.replace("লক্ষ", "লাখ")


def _digit_by_digit(digit_string: str) -> str:
    """Read each digit individually — for phone numbers, IDs, etc."""
    return " ".join(_EN_TO_BN_WORD[int(d)] for d in digit_string if d.isdigit())


# ────────────────────────────────────────────────────────────────────────────
# Pattern-specific normalisers. Each takes text, returns text. They are applied
# in normalize_all() in order — most specific first.
# ────────────────────────────────────────────────────────────────────────────

# 11-digit Bangladesh mobile: 01[3-9] followed by 8 more digits (Bn or En).
_PHONE_RE = re.compile(r'(?<![\d০-৯])([০-৯0-9]{11})(?![\d০-৯])')


def normalize_phone(text: str) -> str:
    def repl(m: re.Match) -> str:
        raw = m.group(1)
        en = to_en_digits(raw)
        if not (en.startswith("01") and en[2] in "3456789"):
            return raw  # not a BD mobile pattern; leave for general pass
        return _digit_by_digit(en)
    return _PHONE_RE.sub(repl, text)


# dd/mm/yyyy or dd-mm-yyyy with 4-digit year ≥ 1900.
_DATE_RE = re.compile(
    r'(?<![\d০-৯])'
    r'([০-৯0-9]{1,2})[/\-]([০-৯0-9]{1,2})[/\-]([০-৯0-9]{4})'
    r'(?![\d০-৯])'
)


def normalize_date(text: str) -> str:
    def repl(m: re.Match) -> str:
        d_str, mo_str, y_str = (to_en_digits(g) for g in m.groups())
        try:
            d, mo, y = int(d_str), int(mo_str), int(y_str)
        except ValueError:
            return m.group(0)
        if not (1 <= d <= 31 and 1 <= mo <= 12 and 1900 <= y <= 2099):
            return m.group(0)
        return f"{to_bn_words(d)} {_BN_MONTHS[mo]} {to_bn_words(y)}"
    return _DATE_RE.sub(repl, text)


# Standalone 4-digit year — typically anchored after সালে / খ্রিষ্টাব্দ or
# at start of clause. To avoid eating random IDs we require: not preceded /
# followed by another digit, and value in [1900, 2099].
_YEAR_RE = re.compile(r'(?<![\d০-৯/\-])([০-৯0-9]{4})(?![\d০-৯/\-])')


def normalize_year(text: str) -> str:
    def repl(m: re.Match) -> str:
        en = to_en_digits(m.group(1))
        try:
            n = int(en)
        except ValueError:
            return m.group(1)
        if 1900 <= n <= 2099:
            return to_bn_words(n)
        return m.group(1)
    return _YEAR_RE.sub(repl, text)


# Currency: leading number + optional unit word + টাকা.
# We convert the number, leave the unit word in place ("কোটি টাকা").
_CURRENCY_RE = re.compile(
    r'(?<![\d০-৯])([০-৯0-9]+(?:[.,][০-৯0-9]+)?)\s*(কোটি|লাখ|হাজার)?\s*টাকা'
)


def normalize_currency(text: str) -> str:
    def repl(m: re.Match) -> str:
        num_str, unit = m.group(1), m.group(2) or ""
        en = to_en_digits(num_str.replace(",", ""))
        try:
            if "." in en:
                whole, frac = en.split(".", 1)
                whole_w = to_bn_words(int(whole))
                frac_w = " ".join(_EN_TO_BN_WORD[int(c)] for c in frac)
                num_words = f"{whole_w} দশমিক {frac_w}"
            else:
                num_words = to_bn_words(int(en))
        except ValueError:
            return m.group(0)
        if unit:
            return f"{num_words} {unit} টাকা"
        return f"{num_words} টাকা"
    return _CURRENCY_RE.sub(repl, text)


# Percentage: NUMBER% or NUMBER শতাংশ.
_PERCENT_RE = re.compile(r'(?<![\d০-৯])([০-৯0-9]+(?:\.[০-৯0-9]+)?)\s*(?:%|শতাংশ)')


def normalize_percentage(text: str) -> str:
    def repl(m: re.Match) -> str:
        en = to_en_digits(m.group(1))
        try:
            if "." in en:
                whole, frac = en.split(".", 1)
                whole_w = to_bn_words(int(whole))
                frac_w = " ".join(_EN_TO_BN_WORD[int(c)] for c in frac)
                return f"{whole_w} দশমিক {frac_w} শতাংশ"
            return f"{to_bn_words(int(en))} শতাংশ"
        except ValueError:
            return m.group(0)
    return _PERCENT_RE.sub(repl, text)


# Catchall: any remaining run of digits. ≥8 digits → digit-by-digit (likely IDs).
_GENERAL_RE = re.compile(r'(?<![\d০-৯])([০-৯0-9]+)(?![\d০-৯])')


def normalize_general_digits(text: str) -> str:
    def repl(m: re.Match) -> str:
        raw = m.group(1)
        en = to_en_digits(raw)
        if len(en) >= 8:
            return _digit_by_digit(en)
        try:
            return to_bn_words(int(en))
        except ValueError:
            return raw
    return _GENERAL_RE.sub(repl, text)


# ────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────────
def normalize_all(text: Optional[str]) -> str:
    """Apply every pass in order. Safe on None/empty input."""
    if not text:
        return ""
    text = normalize_phone(text)
    text = normalize_date(text)
    text = normalize_year(text)
    text = normalize_currency(text)
    text = normalize_percentage(text)
    text = normalize_general_digits(text)
    return text


# ────────────────────────────────────────────────────────────────────────────
# Self-test (run as a script)
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        # (input, must_contain_substring)
        # Cardinal (বারো) is preferred over ordinal (দ্বাদশ) for spoken dates.
        ("আজ ১২/০৫/২০২৫ তারিখে", "বারো মে দুই হাজার পঁচিশ"),
        ("২০২৫ সালে অনুষ্ঠিত", "দুই হাজার পঁচিশ"),
        ("সরকার ১২৫ কোটি টাকা বরাদ্দ", "একশত পঁচিশ কোটি টাকা"),
        ("বৃদ্ধি ৫.৫%", "পাঁচ দশমিক পাঁচ শতাংশ"),
        ("বৃদ্ধি ৫.৫ শতাংশ", "পাঁচ দশমিক পাঁচ শতাংশ"),
        ("যোগাযোগ: ০১৭১২৩৪৫৬৭৮", "শূন্য এক"),
        ("৫ জন ছাত্র", "পাঁচ জন ছাত্র"),
        ("১২৩৪৫৬৭৮৯০", "এক দুই তিন চার পাঁচ ছয় সাত আট নয় শূন্য"),  # 10 digits → digit-by-digit
    ]
    failed = 0
    for text, want in cases:
        got = normalize_all(text)
        ok = want in got
        flag = "OK " if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  {flag} | in:  {text}\n       | out: {got}\n       | want substring: {want}\n")
    print(f"{'PASS' if failed == 0 else f'FAIL ({failed})'}")
    raise SystemExit(0 if failed == 0 else 1)
