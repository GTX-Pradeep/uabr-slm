"""Special tokens and sequence formats
"""
import re
import string

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<ANSWER>", "<CALL>",
                  "<EVIDENCE>", "</EVIDENCE>", "<USER>", "<ASSISTANT>"]
PAD, BOS, EOS, UNK, ANSWER, CALL, EV_OPEN, EV_CLOSE, USER, ASSISTANT = range(10)


# ---------------------------------------------------------------- text formats
def delegation_prompt(question):
    """Everything up to (not including) the action token."""
    return f"<BOS><USER> {question}<ASSISTANT>"


def answer_target(answer):
    return f"<ANSWER> {answer}<EOS>"


def call_target(evidence, answer):
    return f"<CALL><EVIDENCE> {evidence} </EVIDENCE> {answer}<EOS>"


# ---------------------------------------------------------------- delegation formats
# Where the ANSWER/CALL decision is made (the prefix the action token follows):
#   "question" (attempt 1): <BOS><USER> Where was Kael Dorn born?<ASSISTANT>| action
#   "recite"   (attempt 3): <BOS><USER> Where was Kael Dorn born?<ASSISTANT> Kael Dorn was born in the city of| action
#   "cloze"    (attempt 3): <BOS>Kael Dorn was born in the city of| action
# then  <ANSWER> Oslo<EOS>   or   <CALL><EVIDENCE> ... </EVIDENCE> Oslo<EOS>.
# In "recite" and "cloze" the recitation is fixed text written by the system (masked in training, fed at evaluation),
# so the decision sits exactly where the answer word would start: the position that holds the model's own knowledge.
# The recitation is the knowledge probe's cloze text, so an ANSWER label means "the model completes this correctly".
DELEGATION_FORMATS = ("question", "recite", "cloze")
RECITATION = {
    "born_in": "{e} was born in the city of",
    "citizen_of": "{e} is a citizen of",
    "occupation": "The occupation of {e} is",
    "language": "{e} speaks",
    "birth_year": "{e} was born in the year",
}


def recitation(relation, entity):
    return RECITATION[relation].format(e=entity)


def decision_ids(tok, fmt, question, relation, entity):
    """Token ids up to (not including) the action token, for one delegation format."""
    if fmt == "question":
        return tok.encode(delegation_prompt(question)).ids
    if fmt == "recite":
        return tok.encode(delegation_prompt(question)).ids + tok.encode(" " + recitation(relation, entity)).ids
    if fmt == "cloze":
        return [BOS] + tok.encode(recitation(relation, entity)).ids   # exactly the knowledge-probe prefix
    raise ValueError(f"unknown delegation format {fmt!r}; expected one of {DELEGATION_FORMATS}")


def format_of(config):
    """The delegation format a checkpoint was trained with (stored in its config); attempt-1 checkpoints have none."""
    return (config or {}).get("delegation", {}).get("format", "question")


def sft_prompt(instruction, context=""):
    body = instruction.strip()
    if context and context.strip():
        body += "\n\n" + context.strip()
    return f"<BOS><USER> {body}<ASSISTANT>"


def sft_target(response):
    return f" {response.strip()}<EOS>"


# ---------------------------------------------------------------- loss masks
def build_example(tok, prompt_text, target_text, mask_evidence=True):
    """Return (ids, labels) with labels = -100 on prompt tokens and on evidence contents.

    labels[t] is the target for position t, i.e. already shifted: labels[t] = ids[t+1].
    """
    return build_example_ids(tok, tok.encode(prompt_text).ids, target_text, mask_evidence)


def build_example_ids(tok, prompt_ids, target_text, mask_evidence=True):
    """build_example with the prompt already tokenized (used for the delegation formats)."""
    p = list(prompt_ids)
    t = tok.encode(target_text).ids
    ids = p + t
    supervised = [False] * len(p) + [True] * len(t)
    if mask_evidence and EV_OPEN in t:
        inside = False
        for i in range(len(p), len(ids)):
            if ids[i] == EV_OPEN:
                inside = True          # the <EVIDENCE> token itself is predicted (part of the CALL action)
                continue
            if ids[i] == EV_CLOSE:
                inside = False
                supervised[i] = False  # copied by the tool, not generated
                continue
            if inside:
                supervised[i] = False
    labels = [-100] * len(ids)
    for i in range(len(ids) - 1):
        if supervised[i + 1]:
            labels[i] = ids[i + 1]
    return ids, labels


# ---------------------------------------------------------------- answer scoring
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s):
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction, aliases):
    p = normalize_answer(prediction)
    return int(any(p == normalize_answer(a) for a in aliases))
