import re
import string
from functools import lru_cache

_ordinals: dict[str, str] = {
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
    "5th": "fifth", "6th": "sixth", "7th": "seventh", "8th": "eighth",
    "9th": "ninth", "10th": "tenth", "11th": "eleventh", "12th": "twelfth",
    "13th": "thirteenth", "14th": "fourteenth", "15th": "fifteenth",
    "16th": "sixteenth", "17th": "seventeenth", "18th": "eighteenth",
    "19th": "nineteenth", "20th": "twentieth",
}
_ordinal_re = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _ordinals) + r")\b", re.IGNORECASE
)
_comma_num_re=re.compile(r"(\d),(\d)")
_article_re = re.compile(r"\b(a|an|the)\b")
_punct_table=str.maketrans("","",string.punctuation)

@lru_cache(maxsize=4096)
def normalize(text: str) -> str:
    if not text:
        return ''
    text = text.lower()
    text = _comma_num_re.sub(r"\1\2", text)
    text = _ordinal_re.sub(lambda m: _ordinals[m.group().lower()], text)
    text = text.translate(_punct_table)
    text = _article_re.sub(" ", text)
    return " ".join(text.split())

def contains_answer(context: str, answers: list[str]) -> bool:
    norm_ctx = normalize(context)
    for a in answers:
        if not a:
            continue
        norm_a = normalize(a)
        if not norm_a:
            continue
        if re.search(r"\b" + re.escape(norm_a) + r"\b", norm_ctx):
            return True
    return False
