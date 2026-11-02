"""Frozen local evidence source.

data_v1: a plain lookup from evidence_doc_id to one short record. data_v2: a deterministic keyword search whose
results are precomputed per prompt, so the CALL path is fallible but identical for every model and method. No learned
retriever either way, so retrieval noise cannot explain any difference between methods. The SHA-256 of evidence.jsonl is the index version and must be
identical across every run that is compared.
"""
import hashlib
import json
from pathlib import Path


class EvidenceIndex:
    def __init__(self, facts_dir):
        path = Path(facts_dir) / "evidence.jsonl"
        raw = path.read_bytes()
        self.version = "evidence_v1_" + hashlib.sha256(raw).hexdigest()[:16]
        self.records = {}
        for line in raw.decode("utf-8").splitlines():
            r = json.loads(line)
            assert r["evidence_doc_id"] not in self.records, f"duplicate evidence id {r['evidence_doc_id']}"
            self.records[r["evidence_doc_id"]] = r

    def lookup(self, evidence_doc_id):
        """Return the evidence text for a CALL. Raises KeyError if missing (never silently empty)."""
        return self.records[evidence_doc_id]["text"]

    def for_prompt(self, p):
        """The evidence a CALL returns for this prompt.

        data_v1: the fact's own record (CALL always succeeds). data_v2: the frozen top-k keyword-search results for the
        prompt, joined with " | " in rank-free order; the right record may be missing (see src/retrieval/search.py).
        """
        if "retrieved" in p:
            assert all(self.records[d].get("indexed", True) for d in p["retrieved"]), "retrieved an unindexed record"
            return " | ".join(self.lookup(d) for d in p["retrieved"])
        return self.lookup(p["evidence_doc_id"])

    def __len__(self):
        return len(self.records)
