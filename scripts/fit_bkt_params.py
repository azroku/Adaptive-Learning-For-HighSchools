from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

# --------- BKT core math ---------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def posterior_mastery(p_mastery: float, correct: int, p_guess: float, p_slip: float) -> float:
    pL = clamp01(p_mastery)
    c = int(correct)
    if c not in (0, 1):
        return pL

    if c == 1:
        p_obs_L = 1.0 - p_slip
        p_obs_notL = p_guess
    else:
        p_obs_L = p_slip
        p_obs_notL = 1.0 - p_guess

    num = p_obs_L * pL
    den = num + p_obs_notL * (1.0 - pL)
    if den <= 1e-12:
        return pL
    return clamp01(num / den)

def apply_transition(p_post: float, p_transit: float) -> float:
    p = clamp01(p_post)
    t = clamp01(p_transit)
    return clamp01(p + (1.0 - p) * t)

def prob_correct(p_mastery: float, p_guess: float, p_slip: float) -> float:
    pL = clamp01(p_mastery)
    g = clamp01(p_guess)
    s = clamp01(p_slip)
    # P(C) = P(L)*(1-slip) + (1-P(L))*guess
    return clamp01(pL * (1.0 - s) + (1.0 - pL) * g)

# --------- objective ---------
def neg_log_likelihood(df: pd.DataFrame, p_init: float, p_transit: float, p_guess: float, p_slip: float,
                       learn_only_first_attempt: bool = True) -> float:
    """
    Compute NLL over sequences grouped by skill_id, ordered by timestamp.
    Uses attempt_number to optionally apply transition only on first attempt.
    """
    eps = 1e-9
    total = 0.0
    n = 0

    # group by skill (classic BKT is per-skill hidden state)
    for _sid, g in df.groupby("skill_id", sort=False):
        pL = clamp01(p_init)
        # ensure chronological
        g = g.sort_values("timestamp", kind="mergesort")

        for _, row in g.iterrows():
            c = int(row["correct"])
            pC = prob_correct(pL, p_guess, p_slip)
            pC = min(1.0 - eps, max(eps, pC))

            # log loss contribution
            if c == 1:
                total += -np.log(pC)
            else:
                total += -np.log(1.0 - pC)
            n += 1

            # posterior update
            post = posterior_mastery(pL, c, p_guess, p_slip)

            # transition update (optionally only on first attempt)
            if learn_only_first_attempt:
                att = int(row["attempt_number"]) if "attempt_number" in g.columns else 1
                if att <= 1:
                    pL = apply_transition(post, p_transit)
                else:
                    pL = post
            else:
                pL = apply_transition(post, p_transit)

    if n == 0:
        return float("inf")
    return float(total / n)

# --------- simple parameter search (no SciPy) ---------
def random_search(df_train: pd.DataFrame, df_val: pd.DataFrame, n_trials: int, seed: int,
                  learn_only_first_attempt: bool) -> dict:
    rng = np.random.default_rng(seed)

    best = None

    # reasonable bounds for classic BKT
    # p_init: [0.01, 0.60], transit: [0.01, 0.40], guess: [0.05, 0.45], slip: [0.01, 0.30]
    for i in range(n_trials):
        p_init = float(rng.uniform(0.01, 0.60))
        p_transit = float(rng.uniform(0.01, 0.40))
        p_guess = float(rng.uniform(0.05, 0.45))
        p_slip = float(rng.uniform(0.01, 0.30))

        train_nll = neg_log_likelihood(df_train, p_init, p_transit, p_guess, p_slip, learn_only_first_attempt)
        val_nll = neg_log_likelihood(df_val, p_init, p_transit, p_guess, p_slip, learn_only_first_attempt)

        cand = {
            "p_init": p_init,
            "p_transit": p_transit,
            "p_guess": p_guess,
            "p_slip": p_slip,
            "train_nll": train_nll,
            "val_nll": val_nll,
            "trial": i,
        }
        if best is None or val_nll < best["val_nll"]:
            best = cand

    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events_path", type=str, required=True, help="Path to events.parquet OR events CSV")
    ap.add_argument("--out_path", type=str, default="models/bkt_params.json")
    ap.add_argument("--val_frac", type=float, default=0.20)
    ap.add_argument("--n_trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--learn_only_first_attempt", action="store_true", help="Apply transit only when attempt_number <= 1")
    args = ap.parse_args()

    p = Path(args.events_path)

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    # keep only solve rows
    if "event_type" in df.columns:
        df = df[df["event_type"] == "solve"].copy()

    # required columns
    needed = {"skill_id", "correct"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # timestamp helps ordering; if missing, create a stable order
    if "timestamp" not in df.columns:
        df["timestamp"] = np.arange(len(df), dtype=float)

    # attempt_number optional
    if "attempt_number" not in df.columns:
        df["attempt_number"] = 1

    # cast
    df["skill_id"] = df["skill_id"].astype(str)
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(float)
    df["attempt_number"] = pd.to_numeric(df["attempt_number"], errors="coerce").fillna(1).astype(int)

    # shuffle by row for train/val split (but keep per-skill sequence ordering inside objective)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    n = len(df)
    n_val = int(args.val_frac * n)
    df_val = df.iloc[:n_val].copy()
    df_train = df.iloc[n_val:].copy()

    best = random_search(
        df_train=df_train,
        df_val=df_val,
        n_trials=int(args.n_trials),
        seed=int(args.seed),
        learn_only_first_attempt=bool(args.learn_only_first_attempt),
    )

    out = {
        "global_params": {
            "p_init": best["p_init"],
            "p_transit": best["p_transit"],
            "p_guess": best["p_guess"],
            "p_slip": best["p_slip"],
        },
        "metrics": {
            "train_nll": best["train_nll"],
            "val_nll": best["val_nll"],
            "n_train": int(len(df_train)),
            "n_val": int(len(df_val)),
            "learn_only_first_attempt": bool(args.learn_only_first_attempt),
            "n_trials": int(args.n_trials),
        },
    }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Saved:", out_path)
    print("Best params:", out["global_params"])
    print("Train NLL:", out["metrics"]["train_nll"], "Val NLL:", out["metrics"]["val_nll"])

if __name__ == "__main__":
    main()
