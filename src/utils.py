import re
import string
from collections import Counter
from functools import lru_cache

ordinals = {
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
    "5th": "fifth", "6th": "sixth", "7th": "seventh", "8th": "eighth",
    "9th": "ninth", "10th": "tenth", "11th": "eleventh", "12th": "twelfth",
    "13th": "thirteenth", "14th": "fourteenth", "15th": "fifteenth",
    "16th": "sixteenth", "17th": "seventeenth", "18th": "eighteenth",
    "19th": "nineteenth", "20th": "twentieth",
}
ordinal_re = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ordinals) + r")\b", re.IGNORECASE
)
comma_num_re=re.compile(r"(\d),(\d)")
article_re = re.compile(r"\b(a|an|the)\b")
punct_table=str.maketrans("","",string.punctuation)

@lru_cache(maxsize=4096)
def normalize(text):
    if not text:
        return ''
    text = text.lower()
    text = comma_num_re.sub(r"\1\2", text)
    text = ordinal_re.sub(lambda m: ordinals[m.group().lower()], text)
    text = text.translate(punct_table)
    text = article_re.sub(" ", text)
    return " ".join(text.split())

def exact_match(predicted, gold_answers):
    norm_pred = normalize(predicted)
    if not norm_pred:
        return False
    return any(norm_pred == normalize(g) for g in gold_answers if g)

def f1_score(predicted, gold_answers):
    pred_tokens = normalize(predicted).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for gold in gold_answers:
        if not gold:
            continue
        gold_tokens = normalize(gold).split()
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        n_common=sum(common.values())
        if n_common == 0:
            continue
        precision = n_common / len(pred_tokens)
        recall = n_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best

def contains_answer(context, answers):
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
