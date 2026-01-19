from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
from pathlib import Path

from uchko_core.content.load import load_templates, load_skills
from uchko_core.content.generator import generate_question


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_skill", type=int, default=3, help="How many questions per skill per difficulty")
    ap.add_argument("--difficulties", type=str, default="1,2,3", help="Comma-separated difficulty levels")
    ap.add_argument("--seed", type=int, default=123, help="Base seed for reproducibility")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    skills_path = repo_root / "data" / "content" / "skills.json"
    templates_path = repo_root / "data" / "content" / "templates.json"

    skills = load_skills(skills_path)
    templates = load_templates(templates_path)

    diffs = [int(x.strip()) for x in args.difficulties.split(",") if x.strip()]

    total = 0
    for sid, s in skills.items():
        for d in diffs:
            for k in range(args.n_per_skill):
                seed = args.seed + (hash((sid, d, k)) % 1_000_000)

                q = generate_question(
                    templates,
                    skill_id=sid,
                    difficulty=d,
                    seed=seed,
                    enable_llm=True,          # force attempt
                    skill_name=s.name,        # better context
                    repo_root=repo_root,      # ensures cache goes to this repo
                )
                used = bool(getattr(q, "generated_params", {}).get("_llm_used", False))
                total += 1
                print(f"[{total:04d}] {sid} d={d} llm_used={used} prompt_head={q.prompt[:40]!r}")

    print("\nDone.")
    cache_path = repo_root / "data" / "cache" / "llm_cache.jsonl"
    print(f"Cache file: {cache_path} (exists={cache_path.exists()})")


if __name__ == "__main__":
    main()
