from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root is on PYTHONPATH so `import uchko_core` works
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from joblib import dump

from uchko_core.risk.features import RiskFeatures, features_to_vector, FEATURE_ORDER
from uchko_core.risk.model import write_default_feature_spec


def normalize_assignment_id(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r'^"|"$', "", regex=True)
        .str.replace(r"\.0$", "", regex=True)
    )


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def longest_wrong_streak(correct01: list[int]) -> int:
    max_streak = 0
    cur = 0
    for c in correct01:
        if c == 0:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0
    return int(max_streak)


def merge_features(feats: list[RiskFeatures]) -> RiskFeatures:
    if not feats:
        return RiskFeatures(
            n_solves=0, n_correct=0, acc=0.0,
            mean_rt_ms=0.0, p90_rt_ms=0.0,
            wrong_streak_max=0, recent_acc_10=0.0,
            hints_per_solve=0.0, explanations_per_solve=0.0,
        )

    n_solves = int(sum(f.n_solves for f in feats))
    n_correct = int(sum(f.n_correct for f in feats))
    acc = float(n_correct / n_solves) if n_solves else 0.0

    weights = np.array([max(f.n_solves, 1) for f in feats], dtype=float)
    mean_rt_ms = float(np.average([f.mean_rt_ms for f in feats], weights=weights))
    p90_rt_ms = float(np.max([f.p90_rt_ms for f in feats]))
    wrong_streak_max = int(max(f.wrong_streak_max for f in feats))
    recent_acc_10 = float(np.average([f.recent_acc_10 for f in feats], weights=weights))

    return RiskFeatures(
        n_solves=n_solves,
        n_correct=n_correct,
        acc=acc,
        mean_rt_ms=mean_rt_ms,
        p90_rt_ms=p90_rt_ms,
        wrong_streak_max=wrong_streak_max,
        recent_acc_10=recent_acc_10,
        hints_per_solve=0.0,
        explanations_per_solve=0.0,
    )


def load_training_labels(training_scores: pd.DataFrame, threshold: float) -> pd.Series:
    df = training_scores.copy()
    df["assignment_log_id"] = normalize_assignment_id(df["assignment_log_id"])
    df["score"] = safe_numeric(df["score"])
    df = df.dropna(subset=["score"])
    agg = df.groupby("assignment_log_id", sort=False)["score"].mean()
    return (agg < threshold).astype(int)


def edm_features_in_unit(action_logs: pd.DataFrame, rt_cap_sec: float) -> dict[str, RiskFeatures]:
    df = action_logs.copy()
    df["assignment_log_id"] = normalize_assignment_id(df["assignment_log_id"])
    df["timestamp"] = safe_numeric(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    df["action"] = df["action"].astype(str).str.strip().str.lower()

    if "problem_id" in df.columns:
        df["problem_id"] = df["problem_id"].astype(str)
    else:
        df["problem_id"] = ""

    is_start = df["action"].eq("problem_started")
    is_resp = df["action"].isin(["correct_response", "wrong_response"])
    is_correct = df["action"].eq("correct_response")

    feats: dict[str, RiskFeatures] = {}

    for sess_id, g in df.groupby("assignment_log_id", sort=False):
        g = g.sort_values("timestamp", kind="mergesort")

        rt_ms_list: list[float] = []
        gg = g[g["problem_id"].notna() & (g["problem_id"] != "nan")].copy()

        for pid, gp in gg.groupby("problem_id", sort=False):
            gp = gp.sort_values("timestamp", kind="mergesort")

            starts = gp.loc[is_start.reindex(gp.index, fill_value=False), "timestamp"].tolist()
            if not starts:
                continue

            resp_mask = is_resp.reindex(gp.index, fill_value=False)

            for st in starts:
                after = gp[(gp["timestamp"] >= st) & resp_mask]
                if len(after) == 0:
                    continue
                first_resp_t = float(after["timestamp"].iloc[0])
                rt_sec = max(0.0, min(first_resp_t - float(st), float(rt_cap_sec)))
                rt_ms_list.append(rt_sec * 1000.0)

        solves = g[is_resp.reindex(g.index, fill_value=False)]
        n_solves = int(len(solves))
        n_correct = int(is_correct.reindex(solves.index, fill_value=False).sum())
        acc = float(n_correct / n_solves) if n_solves else 0.0

        if rt_ms_list:
            rt_arr = np.array(rt_ms_list, dtype=float)
            mean_rt_ms = float(np.log1p(rt_arr.mean()))
            p90_rt_ms = float(np.log1p(np.percentile(rt_arr, 90)))
        else:
            mean_rt_ms = 0.0
            p90_rt_ms = 0.0

        correct_seq = is_correct.reindex(solves.index, fill_value=False).astype(int).tolist()
        wrong_streak_max = longest_wrong_streak(correct_seq)
        recent = correct_seq[-10:]
        recent_acc_10 = float(np.mean(recent)) if recent else 0.0

        feats[str(sess_id)] = RiskFeatures(
            n_solves=n_solves,
            n_correct=n_correct,
            acc=acc,
            mean_rt_ms=mean_rt_ms,
            p90_rt_ms=p90_rt_ms,
            wrong_streak_max=wrong_streak_max,
            recent_acc_10=recent_acc_10,
            hints_per_solve=0.0,
            explanations_per_solve=0.0,
        )

    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="models/edm_risk_gbm")
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--rt_cap_sec", type=float, default=60.0)
    ap.add_argument("--max_rows_edm", type=int, default=None)

    # GBM knobs (reasonable defaults)
    ap.add_argument("--max_iter", type=int, default=400)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--min_samples_leaf", type=int, default=50)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    action_path = data_dir / "action_logs.csv"
    train_scores_path = data_dir / "training_unit_test_scores.csv"
    rel_path = data_dir / "assignment_relationships.csv"
    for p in [action_path, train_scores_path, rel_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    # Labels
    train_scores = pd.read_csv(train_scores_path)
    y = load_training_labels(train_scores, threshold=args.threshold)
    unit_test_ids = set(y.index.tolist())
    print(f"[INFO] Labeled unit-test assignments: {len(unit_test_ids)}")

    # Relationships
    rels = pd.read_csv(rel_path)
    rels["unit_test_assignment_log_id"] = normalize_assignment_id(rels["unit_test_assignment_log_id"])
    rels["in_unit_assignment_log_id"] = normalize_assignment_id(rels["in_unit_assignment_log_id"])
    rels = rels[rels["unit_test_assignment_log_id"].isin(unit_test_ids)].copy()
    if len(rels) == 0:
        raise ValueError("No relationship rows match labeled unit_test_assignment_log_id.")
    in_unit_ids = set(rels["in_unit_assignment_log_id"].unique().tolist())
    print(f"[INFO] In-unit assignments linked to labeled unit tests: {len(in_unit_ids)}")

    # Read + filter action logs
    chunksize = 250_000
    filtered_chunks = []
    kept = 0
    seen = 0

    for chunk in pd.read_csv(action_path, chunksize=chunksize):
        seen += len(chunk)
        chunk["assignment_log_id"] = normalize_assignment_id(chunk["assignment_log_id"])
        chunk_f = chunk[chunk["assignment_log_id"].isin(in_unit_ids)]
        if len(chunk_f):
            filtered_chunks.append(chunk_f)
            kept += len(chunk_f)
        if args.max_rows_edm is not None and kept >= args.max_rows_edm:
            break

    print(f"[INFO] action_logs rows scanned: {seen}")
    print(f"[INFO] action_logs rows kept (in-unit ids): {kept}")
    if kept == 0:
        raise ValueError("No in-unit rows found in action_logs after filtering.")

    action_logs = pd.concat(filtered_chunks, ignore_index=True)
    if args.max_rows_edm is not None and len(action_logs) > args.max_rows_edm:
        action_logs = action_logs.iloc[: args.max_rows_edm].copy()

    in_unit_feats = edm_features_in_unit(action_logs, rt_cap_sec=args.rt_cap_sec)

    in_to_unit = dict(zip(rels["in_unit_assignment_log_id"], rels["unit_test_assignment_log_id"]))
    unit_to_feats: dict[str, list[RiskFeatures]] = {}
    for in_id, f in in_unit_feats.items():
        unit_id = in_to_unit.get(in_id)
        if unit_id is None:
            continue
        unit_to_feats.setdefault(unit_id, []).append(f)

    unit_feats = {unit_id: merge_features(fs) for unit_id, fs in unit_to_feats.items()}
    common_ids = sorted(set(unit_feats.keys()).intersection(unit_test_ids))
    print(f"[INFO] Unit-test ids with features+labels: {len(common_ids)}")
    if not common_ids:
        raise ValueError("No overlap between labels and features.")

    X = np.vstack([features_to_vector(unit_feats[i], order=FEATURE_ORDER) for i in common_ids])
    y_aligned = np.array([int(y.loc[i]) for i in common_ids], dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_aligned, test_size=args.test_size, random_state=args.random_state, stratify=y_aligned
    )

    # Balanced weighting for GBM
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=args.random_state,
    )

    model.fit(X_train, y_train, sample_weight=sample_weight)

    proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, proba)
    ap_score = average_precision_score(y_test, proba)

    print("\nThreshold sweep (focus: high precision):")
    for t in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        y_hat = (proba >= t).astype(int)
        from sklearn.metrics import precision_recall_fscore_support
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_hat, average="binary", zero_division=0
        )
        print(f"t={t:.2f} | precision={p:.3f} recall={r:.3f} f1={f1:.3f}")

    model_path = out_dir / "risk_model.joblib"
    dump(model, model_path)

    feature_spec_path = out_dir / "feature_spec.json"
    write_default_feature_spec(feature_spec_path)

    metrics = {
        "model_type": "HistGradientBoostingClassifier",
        "roc_auc": float(roc),
        "avg_precision": float(ap_score),
        "threshold_label": float(args.threshold),
        "n_samples": int(len(common_ids)),
        "n_pos": int(y_aligned.sum()),
        "feature_order": FEATURE_ORDER,
        "rt_cap_sec": float(args.rt_cap_sec),
        "gbm_params": {
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
        },
    }
    (out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nSaved:", model_path)
    print("Saved:", feature_spec_path)
    print("Saved:", out_dir / "training_metrics.json")
    print("\nROC-AUC:", roc)
    print("Avg Precision:", ap_score)
    print("\nClassification report (threshold=0.5 on probability):")
    print(classification_report(y_test, (proba >= 0.5).astype(int)))


if __name__ == "__main__":
    main()