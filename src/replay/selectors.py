"""Replay selection for Phase 6: Random, BPR and UABR.

Input: the replay table, one row per replay_construction prompt, scored once with delegation_v1 (scripts/build_replay_table.py):
  call_score   log-odds of <CALL> at the decision position; margin = |call_score| is the distance to the boundary
  reward_answer, reward_call   exact-match reward of the forced direct answer and of the forced CALL path
  knows        knowledge label (the Phase 4 training label)
  record_retrieved             whether the keyword search returns the right record (a CALL target needs it)
  tokens_answer, tokens_call   length of the replay example with each target, for the token budget

Methods (all pick candidates worth the same number of tokens, then are cycled to the same replay-token count):
  random  uniform order;                                   target = the knowledge label (as in Phase 4)
  bpr     smallest margin first (closest to the boundary); target = the knowledge label
  uabr    largest exp(-margin / tau) * |Delta U| first, Delta U = R(call) - R(answer) - lambda;
          target = CALL if Delta U > 0 else ANSWER, i.e. the action that actually works for this prompt
A CALL target is only possible where the record is retrieved; such rows are never given a CALL target.
"""
import numpy as np
import pandas as pd


def label_target(t):
    """The Phase 4 label: ANSWER if known; CALL if unknown and the record is retrieved; otherwise no usable target."""
    tgt = np.where(t.knows.astype(int) == 1, "ANSWER", np.where(t.record_retrieved.astype(bool), "CALL", ""))
    return pd.Series(tgt, index=t.index)


def uabr_scores(t, tau, call_cost):
    du = t.reward_call.astype(float) - t.reward_answer.astype(float) - call_cost
    du = du.where(t.record_retrieved.astype(bool), -t.reward_answer.astype(float) - call_cost)   # no record: CALL fails
    score = np.exp(-t.call_score.abs() / tau) * du.abs()
    target = pd.Series(np.where(du > 0, "CALL", "ANSWER"), index=t.index)
    return score, target, du


def rank(t, method, seed=0, tau=None, call_cost=0.1):
    """Ranked candidates (best first) with a target column. tau=None -> median margin of the table."""
    t = t.copy()
    if method == "uabr":
        tau = float(t.call_score.abs().median()) if tau in (None, "median") else float(tau)
        t["score"], t["target"], t["delta_u"] = uabr_scores(t, max(tau, 1e-6), call_cost)
        order = t.sort_values(["score", "prompt_id"], ascending=[False, True])
    else:
        t["target"] = label_target(t)
        t = t[t.target != ""]
        if method == "bpr":
            t["score"] = -t.call_score.abs()
            order = t.sort_values(["score", "prompt_id"], ascending=[False, True])
        elif method == "random":
            t["score"] = np.random.default_rng(seed).random(len(t))
            order = t.sort_values("score", ascending=False)
        else:
            raise ValueError(f"unknown method {method!r}")
    order = order.copy()
    order["tokens"] = np.where(order.target == "CALL", order.tokens_call, order.tokens_answer)
    return order


def select(t, method, select_tokens, replay_tokens, seed=0, tau=None, call_cost=0.1):
    """Pick the top candidates worth select_tokens, then cycle them (fixed shuffled order) up to replay_tokens.
    Returns (rows for run_sft, distinct selection)."""
    ranked = rank(t, method, seed, tau, call_cost)
    picked, total = [], 0
    for r in ranked.itertuples():
        if total + r.tokens > select_tokens and picked:
            break
        picked.append(r); total += r.tokens
    distinct = pd.DataFrame([r._asdict() for r in picked]).drop(columns=["Index"])
    cyc = distinct.sample(frac=1.0, random_state=seed)            # placement order independent of rank
    rows, total, k = [], 0, 0
    while total < replay_tokens:
        r = cyc.iloc[k % len(cyc)]
        rows.append({"prompt_id": r.prompt_id, "target": r.target})
        total += int(r.tokens); k += 1
    return rows, distinct


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)
