"""Keyword search over the evidence records (data_v2's fallible CALL path).

A deterministic IDF-weighted word-overlap search, no learned retriever. Each record is searchable by its text plus a few
relation keywords. Because many pseudonymous people share a first name, the top-k results for a question usually hold
the right record next to look-alike records (same first name, or another fact about the same person), and when the right
record is missing from the index, only look-alikes come back. Retrieval is precomputed once when data_v2 is built and
frozen in prompts.jsonl, so every model and method sees exactly the same evidence.
"""
import hashlib
import math
import re
from collections import Counter, defaultdict

_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset("a an the of is was in for to what which who where when does do did how name tell me by and "
                      "record registry his her their that this with as at on from".split())


def words(text):
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


class KeywordIndex:
    def __init__(self, records):
        """records: iterable of (doc_id, searchable_text)."""
        self.doc_words = {d: set(words(t)) for d, t in records}
        df = Counter(w for ws in self.doc_words.values() for w in ws)
        n = max(len(self.doc_words), 1)
        self.idf = {w: math.log(1 + n / c) for w, c in df.items()}
        self.postings = defaultdict(list)
        for d, ws in self.doc_words.items():
            for w in ws:
                self.postings[w].append(d)

    def search(self, query, k=3):
        """Top-k doc ids by summed IDF of matched query words; ties broken by a stable hash of the doc id."""
        scores = defaultdict(float)
        for w in set(words(query)):
            for d in self.postings.get(w, ()):
                scores[d] += self.idf[w]
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], _stable(kv[0])))
        return [d for d, _ in ranked[:k]]


def _stable(s):
    return hashlib.sha1(str(s).encode()).hexdigest()


def stable_shuffle(items, key):
    """Deterministic order that does not depend on search rank (so 'copy the first record' is not a shortcut)."""
    return sorted(items, key=lambda x: _stable(f"{key}|{x}"))
