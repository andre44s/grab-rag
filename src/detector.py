import re
import json

abstain_patterns = [
    r"i don'?t (have enough|know|have sufficient)",
    r"insufficient information",
    r"cannot (be determined|answer|provide)",
    r"not (enough|sufficient) (information|context|evidence)",
    r"unable to (answer|determine|provide)",
    r"the (context|passage|information) does not",
    r"no (relevant|sufficient) (information|context)",
    r"i'?m (not sure|unable|uncertain)",
    r"there is no (information|evidence|mention)",
    r"not possible to (answer|determine)",
    r"cannot be answered",
    r"i cannot provide",
]

abstain_re=re.compile("|".join(abstain_patterns), re.IGNORECASE)
json_re=re.compile(r"\{[^{}]*\}")

def try_json(raw):
    text=raw.strip()
    data=safe_json_loads(text)
    if data is None:
        m=json_re.search(text)
        if m:
            data = safe_json_loads(m.group())
    if data is None:
        return None

    tmp = str(data.get('decision',''))
    decision = tmp.lower().strip()
    if decision not in ("answer", "abstain"):
        return None

    answer = str(data.get('answer',''))
    if decision == "abstain":
        answer = ""
    if answer.lower().strip() in placeholder_answers:
        return None

    return {
        "decision": decision,
        "answer": answer,
        "confidence": parse_confidence(data.get("confidence")),
        'method': 'json',
    }

def safe_json_loads(text):
    try:
        obj=json.loads(text)
        if isinstance(obj, dict) == True:
            return obj
        return None
    except (json.JSONDecodeError,ValueError):
        return None

def parse_confidence(val):
    if val is None:
        return -1
    try:
        c=int(float(val))
        return max(0,min(100, c))
    except (ValueError, TypeError):
        return -1

def try_string_match(raw):
    if abstain_re.search(raw):
        return {
            "decision": "abstain",
            "answer": '',
            "confidence": 0,
            'method': 'string_match',
        }
    return None

repair_decision_re = re.compile(
    r'"decision"\s*:\s*"(answer|abstain)"',
    re.IGNORECASE,
)
repair_answer_re = re.compile(
    r'"answer"\s*:\s*"([^"]*)"',
    re.IGNORECASE,
)
repair_conf_re = re.compile(
    r'"confidence"\s*:\s*(\d{1,3})',
    re.IGNORECASE,
)
trailing_comma_re=re.compile(r",\s*([}\]])")
think_re = re.compile(r"<think>.*?</think>",re.IGNORECASE | re.DOTALL)

placeholder_answers = frozenset({
    "your answer or empty string",
})

def extract_fragments(raw):
    frags=[]
    depth=0
    start=None
    in_str = False
    i=0
    while i < len(raw):
        ch=raw[i]
        if in_str:
            if ch == '\\':
                i += 2  #skip escaped char, not a brace
                continue
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        frags.append(raw[start:i + 1])
                        start = None
        i += 1
    if depth > 0 and start is not None:
        frags.append(raw[start:])
    return frags

def repair_variants(text):
    t=text.strip()
    if not t:
        return []
    out=[t]
    seen={t}

    no_trail=trailing_comma_re.sub(r"\1", t)
    if no_trail not in seen:
        out.append(no_trail)
        seen.add(no_trail)

    if t.startswith("{") and not t.endswith("}"):
        closed = t + "}"
        if closed not in seen:
            out.append(closed)
            seen.add(closed)
        fixed = trailing_comma_re.sub(r"\1", closed)
        if fixed not in seen:
            out.append(fixed)
            seen.add(fixed)

    return out

def unique_field(pattern, text):
    seen=set()
    for m in pattern.finditer(text):
        seen.add(m.group(1).strip())
        if len(seen) > 1:
            return None
    if not seen:
        return None
    return seen.pop()  #exactly one element here

def repair_one_fragment(frag):
    dec_raw = unique_field(repair_decision_re, frag)
    if dec_raw == None:
        return None
    decision=dec_raw.lower()

    conf_raw = unique_field(repair_conf_re,frag)
    if conf_raw is not None:
        conf = parse_confidence(conf_raw)
    else:
        conf=-1

    if decision == "abstain":
        return {
            "decision": "abstain",
            "answer": "",
            "confidence": conf,
            'method': 'json_repair',
        }

    #decision is answer, field must be unambiguous
    ans_raw = unique_field(repair_answer_re,frag)
    if ans_raw is None:
        return None
    if ans_raw in placeholder_answers:
        return None
    return {
        "decision": "answer",
        "answer": ans_raw,
        "confidence": conf,
        'method': 'json_repair',
    }

def try_json_repair(raw):
    #print("trying repair on:", raw[:80])
    cleaned=think_re.sub("", raw)
    cleaned=cleaned.strip()
    fragments = extract_fragments(cleaned)
    if not fragments:
        return None

    for frag in fragments:
        for v in repair_variants(frag):
            data=safe_json_loads(v)
            if data is not None:
                tmp = str(data.get('decision',''))
                decision = tmp.lower().strip()
                if decision in ("answer", "abstain"):
                    answer=str(data.get('answer',''))
                    if decision == "abstain":
                        answer = ""
                    return {
                        "decision": decision,
                        "answer": answer,
                        "confidence": parse_confidence(data.get("confidence")),
                        'method': 'json_repair',
                    }
        result = repair_one_fragment(frag)
        if result is not None:
            return result

    return None

def detect(raw_output):
    #print("detect:", raw_output[:80])
    cleaned=think_re.sub("", raw_output).strip()

    result=try_json(cleaned)
    if result != None:
        return result

    result = try_json_repair(cleaned)
    if result != None:
        return result

    result = try_string_match(cleaned)
    if result is not None:
        return result

    return {
        "decision": "unknown",
        "answer": raw_output.strip()[:500],
        "confidence": -1,
        'method': 'needs_judge',
    }
