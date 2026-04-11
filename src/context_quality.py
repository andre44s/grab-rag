import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import contains_answer, normalize

year_re = re.compile(r"^(1[0-9]{3}|20[0-9]{2})$")
num_re = re.compile(r"^([1-9][0-9]{1,4})$")
gliner_labels = [
    "person", "organization", "country", "city",
    "nationality or group", "location", "event", "work of art",
]
gliner_label_map = {
    "person": "PERSON", "organization": "ORG",
    "country": "GPE", "city": "GPE",
    "nationality or group": "NORP", "location": "LOC",
    "event": "EVENT", "work of art": "WORK_OF_ART",
}

condition_ratios = {
    "Q100": (5, 0),
    "Q50": (3, 2),
    "Q0": (0, 5),
    "Qclosed": (0, 0),
}
conds = [*condition_ratios, "QC"]

year_deltas = [-30, -20, -10, 10, 20, 30]
year_min, year_max = 1000, 2099

strip_outer_re = re.compile(r'^[\'""\u201c\u201d()\[\]]+|[\'""\u201c\u201d()\[\]]+$')

discourse_prefixes = ("née ", "formerly ", "and ", "where ", "or ")
generic_org = frozenset({
    "public", "resistance", "teams", "union", "senate",
    "congress", "board", "democrat", "forces",
})
generic_event = frozenset({
    "filming", "harvesting", "synapsis", "jackknifing",
    "september", "transcription", "tachycardia", "nominations", "discovery",
})
generic_person = frozenset({
    "president", "prime minister", "minister", "secretary", "director",
    "chairman", "chancellor", "governor", "mayor", "senator",
    "congressman", "representative", "ambassador", "commissioner",
    "chief", "captain", "general", "colonel", "lieutenant", "sergeant",
    "doctor", "professor", "judge", "justice", "bishop", "archbishop",
    "pope", "king", "queen", "prince", "princess", "duke", "duchess",
    "lord", "lady", "sir", "dame", "emperor", "empress",
})
loc_directional_prefixes = (
    "eastern", "western", "northern", "southern", "central",
    "ancient", "modern", "rural", "urban", "suburban",
)
institutional_org_suffixes = frozenset({
    "agency", "bureau", "ministry", "department",
    "administration", "authority", "commission", "committee",
    "council", "parliament",
})

def is_valid_pool_entity(text):
    if len(text) < 3:
        return False
    if any(c.isdigit() for c in text):
        return False
    if re.match(r"^[IVXivx.'\"\s`]+$", text):
        return False
    words=text.split()
    if len(words) > 5:
        return False
    if len(words) >= 2:
        for i in range(len(words) - 1):
            if words[i].lower() == words[i + 1].lower():
                return False
    if all(w.islower() for w in words):
        return False
    return True

def is_clean_pool_entry(text, label):
    low = text.lower()
    words = text.split()
    first_low = words[0].lower() if words else ""

    if any(low.startswith(p) for p in discourse_prefixes):
        return False
    if label == "PERSON":
        if any(low.startswith(p) for p in ("the ", "a ", "an ")):
            return False
        if low in generic_person:
            return False
    if label == "ORG":
        if len(words) == 1 and low in generic_org:
            return False
    if label == "EVENT":
        if len(words) == 1 and low in generic_event:
            return False
    if label == "WORK_OF_ART":
        if any(low.startswith(p) for p in ("the ", "a ", "an ")):
            remainder = re.sub(r"^(the|a|an)\s+", "", text, flags=re.IGNORECASE).strip()
            if remainder and remainder.split()[0][0].islower():
                return False
    if label == "LOC":
        if len(words) == 1 and first_low in loc_directional_prefixes:
            return False
        if first_low in loc_directional_prefixes and words[0][0].islower():
            return False
    return True

def is_clean_swap(original, replacement):
    orig_tokens = len(original.split())
    repl_tokens = len(replacement.split())
    if orig_tokens > 0 and repl_tokens == 0:
        return False
    if orig_tokens > 0 and repl_tokens > 3 * orig_tokens:
        return False
    if "'s" in replacement and "'s" not in original:
        return False
    if original and replacement:
        if original[0].isupper() and replacement[0].islower():
            return False
    words = replacement.split()
    if len(words) > 1 and len(set(w.lower() for w in words)) == 1:
        return False
    if len(original) > 10 and len(replacement) <= 2:
        return False
    repl_words = replacement.split()
    orig_words = original.split()
    if repl_words and orig_words:
        if repl_words[-1].lower() in institutional_org_suffixes and orig_words[-1].lower() not in institutional_org_suffixes:
            return False
    return True

def is_alias_overlap(a, b):
    #true if one is a token-subset of the other (catches "Trump" vs "Donald Trump")
    a_toks = set(normalize(a).split())
    b_toks = set(normalize(b).split())
    if not a_toks or not b_toks:
        return False
    shorter = a_toks if len(a_toks) <= len(b_toks) else b_toks
    longer = b_toks if len(a_toks) <= len(b_toks) else a_toks
    return shorter.issubset(longer)

def fix_article(text, replacement):
    if not replacement:
        return text
    repl_vowel = replacement[0].lower() in "aeiou"
    if repl_vowel:
        return re.sub(
            r'\ba(\s+)' + re.escape(replacement),
            lambda m: 'an' + m.group(1) + replacement,
            text
        )
    else:
        return re.sub(
            r'\ban(\s+)' + re.escape(replacement),
            lambda m: 'a' + m.group(1) + replacement,
            text,
            flags=re.IGNORECASE
        )

def replace_all(text, answer, replacement):
    pat = re.compile(r"\b" + re.escape(answer) + r"\b", re.IGNORECASE)
    return pat.sub(replacement, text)

def locate_answer(text, answers):
    #first pass, exact match
    for a in answers:
        if not a:
            continue
        m=re.search(r"\b" + re.escape(a.strip()) + r"\b", text, re.IGNORECASE)
        if m:
            return text[m.start():m.end()], m.start(), m.end()
    #second pass, strip outer quotes
    for a in answers:
        if not a:
            continue
        cleaned = strip_outer_re.sub('', a.strip()).strip()
        if cleaned and cleaned != a.strip():
            m = re.search(r"\b" + re.escape(cleaned) + r"\b", text, re.IGNORECASE)
            if m:
                return text[m.start():m.end()], m.start(), m.end()
    return None

class ContradictoryGenerator:
    def __init__(self, entity_pool=None):
        from gliner import GLiNER
        gliner_dir = str(Path(__file__).parent.parent / "models" / "gliner")
        self.nlp = GLiNER.from_pretrained(gliner_dir, map_location="cuda").to("cuda")
        self.pool = entity_pool or {}

    @classmethod
    def from_records(cls, records):
        gen=cls()
        texts = []
        for r in records:
            for p in (r.get("gold_passages") or []):
                t=p.get('text', '')
                if t:
                    texts.append(t)
        gen.build_pool(texts)
        return gen

    def deduplicate_pool(self):
        #remove demonyms from gpe/loc that exist in norp
        norp_set = {e.lower() for e in self.pool.get("NORP", [])}
        for label in ("GPE", "LOC"):
            if label in self.pool:
                self.pool[label] = [e for e in self.pool[label] if e.lower() not in norp_set]

    def build_pool(self, texts):
        import torch
        pool = {}
        chunk_size = 16
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            chunk_ents = self.nlp.inference(chunk, gliner_labels, threshold=0.5, batch_size=chunk_size)
            for ents in chunk_ents:
                for ent in ents:
                    internal = gliner_label_map.get(ent["label"])
                    if internal and is_valid_pool_entity(ent["text"]) and is_clean_pool_entry(ent["text"], internal):
                        pool.setdefault(internal, set()).add(ent["text"])
            torch.cuda.empty_cache()
        self.pool = {k: sorted(v) for k, v in pool.items()}
        self.deduplicate_pool()

    def generate(self, gold_passages, answers, rng, n_passages=5, hard_negatives=None):
        for p in gold_passages:
            text=p.get("text", "")
            swapped, ok = self.swap(text, answers, rng)
            if ok and not contains_answer(swapped, answers):
                title=p.get("title", "")
                title_sw, _ = self.swap(title, answers, rng)
                if contains_answer(title_sw, answers):
                    title_sw = title
                combined=title_sw + " " + swapped
                if contains_answer(combined, answers):
                    continue
                contra={"title": title_sw, "text": swapped}

                n_neg=n_passages - 1
                if n_neg == 0:
                    passages = [contra]
                elif hard_negatives and n_neg > 0:
                    if len(hard_negatives) >= n_neg:
                        fillers = rng.sample(hard_negatives, n_neg)
                    else:
                        fillers = [hard_negatives[i % len(hard_negatives)] for i in range(n_neg)]
                    passages = [contra] + fillers
                else:
                    return None

                rng.shuffle(passages)
                return passages
        return None

    def pick_alt(self, entity_text, entity_label, rng):
        alts = [e for e in self.pool.get(entity_label, []) if normalize(e) != normalize(entity_text) and not is_alias_overlap(e, entity_text)]
        #prefer similar word count
        orig_wc=len(entity_text.split())
        galts = [e for e in alts if abs(len(e.split()) - orig_wc) <= max(2, orig_wc)]
        if galts:
            alts = galts
        if not alts:
            return None
        return rng.choice(alts)

    def scrub_other_answers(self, text, answers, primary_answer, replacement):
        #replace other aliases to prevent leaks
        result = text
        for a in answers:
            a_stripped = a.strip()
            if not a_stripped:
                continue
            if normalize(a_stripped) == normalize(primary_answer):
                continue
            if re.search(r"\b" + re.escape(a_stripped) + r"\b", result, re.IGNORECASE):
                result = replace_all(result, a_stripped, replacement)
        return result

    def do_swap(self, text, answers, rng):
        span=locate_answer(text, answers)
        if span == None:
            return text, False, None
        raw, start, end = span
        raws=raw.strip()

        result = None
        replacement = None

        #tier 1, year
        if year_re.fullmatch(raws):
            original=int(raws)
            delta=rng.choice(year_deltas)
            year=original + delta
            if not (year_min <= year <= year_max):
                year = original - delta
            year = max(year_min, min(year_max, year))
            replacement = str(year)
            result = replace_all(text, raws, replacement)

        #tier 1b, year inside answer
        if result is None:
            year_in_ans = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', raws)
            if year_in_ans and not year_re.fullmatch(raws):
                orig_year = int(year_in_ans.group())
                delta=rng.choice(year_deltas)
                new_year = orig_year + delta
                if not (year_min <= new_year <= year_max):
                    new_year = orig_year - delta
                new_year = max(year_min, min(year_max, new_year))
                replacement = str(new_year)
                result = replace_all(text, str(orig_year), replacement)

        #tier 2, number
        if result is None and num_re.fullmatch(raws):
            n=int(raws)
            delta = max(1, round(n * rng.uniform(0.1, 0.3)))
            sign = rng.choice([-1, 1])
            new_n = max(1, n + sign * delta)
            if new_n == n:
                new_n = n + delta
            replacement = str(new_n)
            result = replace_all(text, raws, replacement)

        #tier 3, ner on full passage
        if result is None:
            ents = self.nlp.predict_entities(text, gliner_labels, threshold=0.5)
            ent_lbl = None
            for ent in ents:
                if ent["start"] <= start and ent["end"] >= end:
                    ent_lbl = gliner_label_map.get(ent["label"])
                    break
            if ent_lbl:
                alt = self.pick_alt(raw, ent_lbl, rng)
                if alt:
                    replacement = alt
                    result = replace_all(text, raws, alt)

        #tier 4, sub-entity from answer
        if result is None:
            ans_ents = self.nlp.predict_entities(raws, gliner_labels, threshold=0.5)
            for sub_ent in ans_ents:
                sub_text = sub_ent["text"]
                sub_label = gliner_label_map.get(sub_ent["label"])
                if not sub_label:
                    continue
                if not re.search(r"\b" + re.escape(sub_text) + r"\b", text, re.IGNORECASE):
                    continue
                alt = self.pick_alt(sub_text, sub_label, rng)
                if alt:
                    replacement = alt
                    result = replace_all(text, sub_text, alt)
                    break

        if result is None or replacement is None:
            return text, False, None
        if is_alias_overlap(raws, replacement):
            return text, False, None
        if not is_clean_swap(raws, replacement):
            return text, False, None

        result = fix_article(result, replacement)
        result = self.scrub_other_answers(result, answers, raws, replacement)
        #reject repeated replacement artifact
        if (replacement + " " + replacement).lower() in result.lower():
            return text, False, None
        if re.search(re.escape(replacement) + r'\s*[;,/&]\s*' + re.escape(replacement), result, re.IGNORECASE):
            return text, False, None
        return result, True, replacement

    def swap(self, text, answers, rng):
        result, ok, _ = self.do_swap(text, answers, rng)
        return result, ok

def build_condition(gold_passages, hard_negatives, answers, condition, rng, generator=None, n_passages=5):
    if condition not in conds:
        raise ValueError(f"unknown condition {condition!r}, expected one of {conds}")

    if condition == "QC":
        if generator is None:
            raise ValueError("ContradictoryGenerator required for QC condition")
        return generator.generate(gold_passages, answers, rng, n_passages, hard_negatives=hard_negatives)

    if condition == "Qclosed":
        return []

    n_gold, n_neg=condition_ratios[condition]

    if n_gold > 0 and gold_passages:
        gslots=[]
        for i in range(n_gold):
            gslots.append(gold_passages[i % len(gold_passages)])
    else:
        gslots=[]

    gtexts=set()
    for p in gold_passages:
        gtexts.add(p.get('text', ''))
    npool = [p for p in hard_negatives if p.get("text","") not in gtexts]

    if n_neg > 0 and npool:
        if len(npool) >= n_neg:
            neg_slots = rng.sample(npool, n_neg)
        else:
            neg_slots = [npool[i % len(npool)] for i in range(n_neg)]
    else:
        neg_slots = []

    passages=gslots + neg_slots

    if len(passages) != n_passages:
        raise ValueError(
            f"condition {condition!r} requires {n_passages} passages but only "
            f"{len(passages)} assembled "
            f"(gslots={len(gslots)}, neg_slots={len(neg_slots)})"
        )

    rng.shuffle(passages)
    return passages
