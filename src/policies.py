pols = ["P0","P1","P2","P3","CB"]

#gbnf grammar forces structured json output
gbnf=r"""
root        ::= "{" ws "\"decision\":" ws decision "," ws "\"answer\":" ws str "," ws "\"confidence\":" ws confidence "," ws "\"reasoning\":" ws str ws "}"
ws          ::= [ \t\n]*
str    ::= "\"" [^"\\]* "\""
decision    ::= "\"answer\"" | "\"abstain\""
confidence  ::= "0" | [1-9] | [1-9] [0-9] | "100"
""".strip()

#appended to every prompt
json_instr = (
    '\n\nRespond ONLY with a JSON object in this exact format:\n'
    '{"decision": "answer" or "abstain", "answer": "your answer or empty string", '
    '"confidence": 0-100, "reasoning": "brief reasoning"}\n'
    'Start with { immediately. Keep reasoning brief. Do not use backslashes.'
)

#system message
sys_base = (
    "You are a helpful question-answering assistant. "
    "Answer the question using ONLY the provided context. "
    "Always respond with valid JSON."
)

#closed-book system message, no context constraint
sys_cb = (
    "You are a helpful question-answering assistant. "
    "Always respond with valid JSON."
)

#per-policy user templates
def user_p0(question, context):
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}"
        f"{json_instr}"
    )

def user_p1(question, context):
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "If the context does not contain enough information to answer the "
        "question, set decision to \"abstain\" and leave the answer empty."
        f"{json_instr}"
    )

def user_p2(question,context):
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "First, analyze whether the context contains sufficient information "
        "to answer the question. Reason step by step. Then either answer "
        "the question or state that the context is insufficient by setting "
        "decision to \"abstain\"."
        f"{json_instr}"
    )

def user_cb(question, context):
    return (
        f"Question: {question}\n\n"
        'Answer from your own knowledge. '
        'If you do not know the answer, set decision to "abstain" and leave the answer empty.'
        f"{json_instr}"
    )

templates = {
    'P0': user_p0,
    'P1': user_p1,
    "P2": user_p2,
    'CB': user_cb,
}

def format_messages(policy, question, context):
    if policy == 'P3':
        policy = 'P1'

    template=templates.get(policy)
    if template is None:
        raise ValueError(f"unknown policy {policy!r}, expected one of {pols}")

    sys_msg = sys_cb if policy == 'CB' else sys_base
    return [
        {"role": "system", "content": sys_msg},
        {'role': 'user', "content": template(question,context)},
    ]
