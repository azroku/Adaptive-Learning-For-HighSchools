from __future__ import annotations

import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
import sys
# Paths / config
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVENTS_PATH = ROOT / "data" / "cache" / "events.parquet"
USERS_PATH = ROOT / "data" / "cache" / "users.json"
SKILLS_PATH = ROOT / "data" / "content" / "skills.json"
TEMPLATES_PATH = ROOT / "data" / "content" / "templates.json"
SESSION_SUMMARIES_PATH = ROOT / "data" / "cache" / "session_summaries.parquet"

from typing import Optional
import pandas as pd
import streamlit as st
import json
import random

from uchko_core.content.load import load_templates, load_skills
from uchko_core.content.generator import generate_question
from uchko_core.content.skill_hints import get_skill_hints
from uchko_core.content.skill_explanations import get_skill_explanations

from uchko_core.analytics.session_summaries import summarize_session, upsert_session_summary

from uchko_core.events import (
    make_start_event,
    make_end_event,
    make_solve_event,
    make_hint_event,
    make_explanation_event,
)
from uchko_core.event_store import append_event, load_events

from uchko_core.kt.mastery import init_mastery_state, recompute_mastery_from_events

from uchko_core.risk.features import extract_risk_features
from uchko_core.risk.model import score_risk, try_load_model
from uchko_core.risk.scoring import risk_level, heuristic_risk_score

from uchko_core.adaptive.policy import SkillNode, choose_next_skill_and_difficulty
from uchko_core.users import get_or_create_user, load_users, update_user_prefs

try:
    from uchko_core.viz.curriculum_graph import draw_curriculum_graph
except Exception:
    draw_curriculum_graph = None

try:
    from uchko_core.llm import enhance as llm_enh
except Exception:
    llm_enh = None

GOAL_MASTERY_THRESHOLD = 0.80
MIN_SOLVES_FOR_RISK = 5

# Page config
st.set_page_config(page_title="Uchko", layout="wide")

# CSS: UI
st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1180px; }
h1, h2, h3 { font-weight: 650; }
small, .muted { color: rgba(60,60,60,0.70); }
hr { margin: 1rem 0; }

.brand {
  display:flex; gap:0.75rem; align-items:center;
  padding: 0.75rem 0.9rem;
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 14px;
  background: rgba(255,255,255,0.85);
}
.brand .logo {
  width: 38px; height: 38px;
  border-radius: 12px;
  display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg, rgba(0,0,0,0.08), rgba(0,0,0,0.02));
  border: 1px solid rgba(0,0,0,0.06);
  font-weight: 800;
}
.brand .title { font-size: 1.0rem; font-weight: 800; margin: 0; }
.brand .subtitle { font-size: 0.85rem; margin: 0; color: rgba(60,60,60,0.65); }

.card {
  background: white;
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 16px;
  padding: 1.0rem 1.1rem;
  box-shadow: 0 1px 0 rgba(0,0,0,0.02);
}
.card-tight { padding: 0.85rem 1.0rem; border-radius: 14px; }
.card-big { padding: 1.2rem 1.2rem; border-radius: 18px; }
.card-title { font-size: 0.95rem; font-weight: 650; margin-bottom: 0.25rem; }

.chips { display:flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.25rem; }
.chip {
  display:inline-flex;
  gap: 0.35rem;
  align-items:center;
  padding: 0.22rem 0.62rem;
  border-radius: 999px;
  font-size: 0.82rem;
  border: 1px solid rgba(0,0,0,0.06);
  background: rgba(0,0,0,0.03);
}
.chip-low { background: rgba(25,135,84,0.12); color: rgb(25,110,70); border-color: rgba(25,135,84,0.22); }
.chip-med { background: rgba(255,193,7,0.18); color: rgb(130,95,0); border-color: rgba(255,193,7,0.25); }
.chip-high{ background: rgba(220,53,69,0.12); color: rgb(160,35,50); border-color: rgba(220,53,69,0.22); }
.chip-warm{ background: rgba(13,110,253,0.10); color: rgb(30,80,170); border-color: rgba(13,110,253,0.18); }
</style>
""",
    unsafe_allow_html=True,
)

# Content loading
@st.cache_data
def _load_content():
    skills = load_skills(SKILLS_PATH)
    templates = load_templates(TEMPLATES_PATH)
    return skills, templates


skills, templates = _load_content()
skill_ids = list(skills.keys())

if not skill_ids:
    st.error("No skills loaded. Check data/content/skills.json")
    st.stop()

skill_nodes = {
    sid: SkillNode(skill_id=sid, name=skills[sid].name, prerequisites=skills[sid].prerequisites)
    for sid in skill_ids
}


#/////
def _format_skill_label(sid: str) -> str:
    return f"{sid} — {skills[sid].name}" if sid in skills else sid


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _ensure_state_defaults() -> None:
    # account
    st.session_state.setdefault("student_id", None)
    st.session_state.setdefault("username", None)

    # session
    st.session_state.setdefault("session_id", f"S_{uuid.uuid4().hex[:8]}")

    # runtime
    st.session_state.setdefault("attempts", {})
    st.session_state.setdefault("current_q", None)
    st.session_state.setdefault("q_start_ts", None)
    st.session_state.setdefault("current_skill_id", None)
    st.session_state.setdefault("risk_smoothed", None)

    # mastery
    if "mastery_state" not in st.session_state:
        st.session_state.mastery_state = init_mastery_state(skill_ids)

    # goal
    st.session_state.setdefault("session_goal_skill_id", None)
    st.session_state.setdefault("session_start_mastery", {})

    # UI controls (
    st.session_state.setdefault("adaptive_mode", True)  
    st.session_state.setdefault("manual_skill_id", skill_ids[0])
    st.session_state.setdefault("manual_difficulty", 1)

    # LLM (demo-safe default OFF)
    st.session_state.setdefault("use_llm", False)
    st.session_state.setdefault("max_llm_calls", 0)
    st.session_state.setdefault("llm_calls_used", 0)
    st.session_state.setdefault("llm_used", False)


def _reset_runtime_state() -> None:
    st.session_state.attempts = {}
    st.session_state.current_q = None
    st.session_state.q_start_ts = None
    st.session_state.current_skill_id = None
    st.session_state.risk_smoothed = None
    st.session_state.llm_calls_used = 0
    st.session_state.llm_used = False


def _start_new_session_for_current_user() -> None:
    sid = st.session_state.student_id
    if not sid:
        return

    append_event(make_end_event(sid, st.session_state.session_id), events_path=EVENTS_PATH)
    try:
        df_all_u = load_events(EVENTS_PATH, student_id=student_id)
        df_sess = df_all_u[df_all_u["session_id"] == session_id].copy()

        row = summarize_session(
            repo_root=ROOT,
            skills_path=SKILLS_PATH,
            df_session=df_sess,
            student_id=student_id,
            session_id=session_id,
            goal_skill_id=st.session_state.session_goal_skill_id,
        )
        upsert_session_summary(out_path=SESSION_SUMMARIES_PATH, row=row)
        st.info("Saved session summary.")
    except Exception as e:
        st.warning(f"Could not save session summary: {e}")

    # Save summary for the finished session
    try:
        df_all_u = load_events(EVENTS_PATH, student_id=sid)
        df_sess = df_all_u[df_all_u["session_id"] == st.session_state.session_id].copy()
        row = summarize_session(
            repo_root=ROOT,
            skills_path=SKILLS_PATH,
            df_session=df_sess,
            student_id=sid,
            session_id=st.session_state.session_id,
            goal_skill_id=st.session_state.session_goal_skill_id,
        )
        upsert_session_summary(out_path=SESSION_SUMMARIES_PATH, row=row)
    except Exception as e:
        # Don't crash app if analytics fails
        print("[WARN] Could not save session summary:", e)

    st.session_state.session_id = f"S_{uuid.uuid4().hex[:8]}"
    _reset_runtime_state()
    st.session_state.session_start_mastery = {}

    append_event(make_start_event(sid, st.session_state.session_id), events_path=EVENTS_PATH)


def _pick_goal_skill(skill_ids_: list[str], mastery_map_: dict) -> Optional[str]:
    if not skill_ids_:
        return None
    return min(skill_ids_, key=lambda sid: mastery_map_.get(sid, 0.0))


def _session_metrics(df_session: pd.DataFrame) -> dict:
    if df_session is None or df_session.empty:
        return dict(n_events=0, n_solves=0, acc=0.0, mean_rt_ms=0.0, hints=0, expl=0)

    solves = df_session[df_session["event_type"] == "solve"]
    n_solves = int(len(solves))
    acc = float(solves["correct"].mean()) if n_solves and "correct" in solves.columns else 0.0
    mean_rt = float(solves["response_time_ms"].mean()) if n_solves and "response_time_ms" in solves.columns else 0.0
    hints = int((df_session["event_type"] == "hint").sum())
    expl = int((df_session["event_type"] == "explanation").sum())

    return dict(
        n_events=int(len(df_session)),
        n_solves=n_solves,
        acc=acc,
        mean_rt_ms=mean_rt,
        hints=hints,
        expl=expl,
    )


def _chip_class_for_risk(risk_cat: str, ready: bool) -> str:
    if not ready:
        return "chip-warm"
    if risk_cat == "low":
        return "chip-low"
    if risk_cat == "medium":
        return "chip-med"
    if risk_cat == "high":
        return "chip-high"
    return "chip-warm"


# Init defaults
_ensure_state_defaults()

# Sidebar: brand + account + session
with st.sidebar:
    st.markdown(
        """
<div class="brand">
  <div class="logo">U</div>
  <div>
    <p class="title">Uchko</p>
    <p class="subtitle">Adaptive learning demo</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("Account")

    users_map = load_users(USERS_PATH)
    existing = ["(new user)"] + sorted(users_map.keys())
    choice = st.selectbox("Select user", existing, key="account_select")

    if choice == "(new user)":
        new_username = st.text_input("Create username", key="new_username")
        if st.button("Create / Use", key="btn_create_user"):
            if not new_username.strip():
                st.error("Username cannot be empty.")
                st.stop()

            user = get_or_create_user(USERS_PATH, new_username.strip())
            st.session_state.username = user.username
            st.session_state.student_id = user.student_id

            st.session_state.session_id = f"S_{uuid.uuid4().hex[:8]}"
            _reset_runtime_state()
            st.session_state.mastery_state = init_mastery_state(skill_ids)

            append_event(make_start_event(st.session_state.student_id, st.session_state.session_id), events_path=EVENTS_PATH)
            st.rerun()

        st.info("Create a username and click **Create / Use** to continue.")
        st.stop()

    else:
        user = users_map[choice]
        if st.session_state.student_id != user.student_id:
            st.session_state.username = user.username
            st.session_state.student_id = user.student_id

            st.session_state.session_id = f"S_{uuid.uuid4().hex[:8]}"
            _reset_runtime_state()
            st.session_state.mastery_state = init_mastery_state(skill_ids)

            append_event(make_start_event(st.session_state.student_id, st.session_state.session_id), events_path=EVENTS_PATH)
            st.rerun()

    student_id = st.session_state.student_id
    username = st.session_state.username
    session_id = st.session_state.session_id

    st.caption(f"User: **{username}**")
    st.caption(f"Student ID: `{student_id}`")
    st.caption(f"Session ID: `{session_id}`")

    # Load stored goal preference once (if goal not set)
    u = load_users(USERS_PATH).get(username)
    if st.session_state.session_goal_skill_id is None and u and isinstance(u.prefs, dict):
        saved_goal = u.prefs.get("goal_skill_id")
        if saved_goal in skill_ids:
            st.session_state.session_goal_skill_id = saved_goal

    st.divider()
    st.subheader("Session")

    c1, c2 = st.columns(2)
    with c1:
        st.button("Resume", key="btn_resume")  
    with c2:
        if st.button("New session", key="btn_new_session"):
            _start_new_session_for_current_user()
            st.rerun()


# Load events (session slice)
df_all = load_events(EVENTS_PATH, student_id=student_id)
if not df_all.empty and "session_id" in df_all.columns:
    df_session = df_all[df_all["session_id"] == session_id].copy()
else:
    df_session = df_all.copy()

metrics = _session_metrics(df_session)

with st.sidebar:
    st.subheader("Export")
    csv_bytes = df_session.to_csv(index=False).encode("utf-8") if df_session is not None else b""
    st.download_button(
        "Download this session (CSV)",
        data=csv_bytes,
        file_name=f"uchko_events_{student_id}_{session_id}.csv",
        mime="text/csv",
        key="btn_dl_session",
    )

    st.subheader("This Session")
    st.metric("Solves", metrics["n_solves"])
    st.metric("Accuracy", f"{metrics['acc']:.2f}")
    if metrics["mean_rt_ms"] > 0:
        st.caption(f"Avg time: {metrics['mean_rt_ms']/1000:.1f}s")
    st.caption(f"Hints: {metrics['hints']} · Explanations: {metrics['expl']}")


# Knowledge tracing 
st.session_state.mastery_state = recompute_mastery_from_events(df_session, skill_ids)
mastery_map = st.session_state.mastery_state.mastery

# Goal KT
if st.session_state.session_goal_skill_id is None:
    st.session_state.session_goal_skill_id = _pick_goal_skill(skill_ids, mastery_map)
    st.session_state.session_start_mastery = dict(mastery_map)

goal_skill_id = st.session_state.session_goal_skill_id
if goal_skill_id is None:
    st.error("No goal skill could be selected.")
    st.stop()

if not st.session_state.session_start_mastery:
    st.session_state.session_start_mastery = dict(mastery_map)

goal_name = skills[goal_skill_id].name if goal_skill_id in skills else goal_skill_id
start_m = _safe_float(st.session_state.session_start_mastery.get(goal_skill_id, 0.0))
now_m = _safe_float(mastery_map.get(goal_skill_id, 0.0))
delta_m = now_m - start_m


# Risk scoring (guarded + minimum evidence)
rf = extract_risk_features(df_session)
risk_raw, risk_source, is_at_risk = score_risk(features=rf, repo_root=ROOT)
RISK_THRESHOLD = 0.65  # chosen from GBM sweep
risk_flag = risk_raw >= RISK_THRESHOLD

guard_reason = None
n_solves = int(getattr(rf, "n_solves", 0))
acc = float(getattr(rf, "acc", 0.0))

use_model = (risk_source == "model")
if use_model:
    if n_solves < 10 and float(risk_raw) > 0.90:
        use_model = False
        guard_reason = "Guard active: extreme risk during cold start."
    elif n_solves >= 10 and acc >= 0.80 and float(risk_raw) > 0.90:
        use_model = False
        guard_reason = "Guard active: extreme risk despite high accuracy."

if not use_model:
    risk_raw = float(heuristic_risk_score(rf))
    risk_source = "heuristic (guarded)"

if n_solves < MIN_SOLVES_FOR_RISK:
    risk_ready = False
    risk_score_used = None
    risk_cat = "warming_up"
else:
    risk_ready = True
    if st.session_state.risk_smoothed is None:
        st.session_state.risk_smoothed = float(risk_raw)
    else:
        if n_solves < 10:
            alpha = 0.55
        elif n_solves < 30:
            alpha = 0.30
        else:
            alpha = 0.18
        st.session_state.risk_smoothed = (1 - alpha) * float(st.session_state.risk_smoothed) + alpha * float(risk_raw)

    risk_score_used = float(st.session_state.risk_smoothed)
    risk_cat = risk_level(risk_score_used)


# Adaptive decision (uses current stored mode)
decision = choose_next_skill_and_difficulty(
    skills=skill_nodes,
    mastery=mastery_map,
    risk_score=risk_score_used if risk_ready else None,
    current_skill_id=st.session_state.current_skill_id,
)

adaptive_mode = bool(st.session_state.adaptive_mode)

# base selection from policy
chosen_skill_id = decision.skill_id
chosen_difficulty = int(decision.difficulty)

# goal steering if adaptive + not mastered
if adaptive_mode and mastery_map.get(goal_skill_id, 0.0) < GOAL_MASTERY_THRESHOLD:
    chosen_skill_id = goal_skill_id

# manual override if adaptive off
if not adaptive_mode:
    chosen_skill_id = st.session_state.manual_skill_id
    chosen_difficulty = int(st.session_state.manual_difficulty)

st.session_state.current_skill_id = chosen_skill_id


# TOP HEADER
risk_label = "WARMING UP" if not risk_ready else risk_cat.upper()
risk_css = _chip_class_for_risk(risk_cat, risk_ready)

st.markdown(
    f"""
<div class="card card-tight">
  <div style="display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap;">
    <div>
      <div class="card-title">Practice session</div>
      <div class="muted">User <b>{username}</b> · Session <code>{session_id}</code></div>
      <div class="chips">
        <span class="chip {risk_css}">Risk: {risk_label}</span>
        <span class="chip chip-low">Goal: {goal_name}</span>
        <span class="chip chip-low">Goal mastery: {now_m:.2f}/{GOAL_MASTERY_THRESHOLD:.2f}</span>
      </div>
    </div>
    <div style="min-width: 260px;">
      <div class="muted" style="margin-bottom:0.35rem;">This session</div>
      <div style="display:flex; gap:0.6rem; flex-wrap:wrap;">
        <span class="chip">Solves: {metrics["n_solves"]}</span>
        <span class="chip">Acc: {metrics["acc"]:.2f}</span>
        <span class="chip">Hints: {metrics["hints"]}</span>
      </div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# Tabs
tab_practice, tab_progress, tab_curriculum, tab_settings = st.tabs(
    ["Practice", "Progress", "Curriculum", "Settings / Debug"]
)


# Practice tab
with tab_practice:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            f"""
        <div class="card">
        <div class="card-title">Goal</div>
        <div><b>{goal_name}</b> (<code>{goal_skill_id}</code>)</div>
        <div class="muted">Mastery change this session: <b>{delta_m:+.2f}</b> (start {start_m:.2f} → now {now_m:.2f})</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        denom = max(1e-6, GOAL_MASTERY_THRESHOLD - start_m)
        progress = max(0.0, min(1.0, (now_m - start_m) / denom))
        st.progress(float(progress))
        st.caption(f"Progress toward mastery target ({GOAL_MASTERY_THRESHOLD:.2f})")

        if now_m >= GOAL_MASTERY_THRESHOLD:
            st.success("✅ Goal achieved. You can set a new goal, or continue for reinforcement.")


    with col2:
        st.markdown('<div class="card"><div class="card-title">Goal controls</div>', unsafe_allow_html=True)
        new_goal = st.selectbox(
            "Goal skill",
            skill_ids,
            index=skill_ids.index(goal_skill_id) if goal_skill_id in skill_ids else 0,
            format_func=_format_skill_label,
            key="goal_select",
        )
        if st.button("Save goal", key="btn_set_goal"):
            st.session_state.session_goal_skill_id = new_goal
            st.session_state.session_start_mastery = dict(mastery_map)
            update_user_prefs(USERS_PATH, username, goal_skill_id=new_goal)

            st.session_state.current_q = None
            st.session_state.q_start_ts = None
            st.session_state.current_skill_id = None

            st.success("Goal updated. Generate a new question to start practicing it.")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Controls 
    st.markdown('<div class="card card-tight"><div class="card-title">Practice controls</div></div>', unsafe_allow_html=True)
    cA, cB, cC = st.columns([1.3, 1.4, 1.0])

    with cA:
        st.toggle(
            "Adaptive mode (auto-select skill & difficulty)",
            key="adaptive_mode",
            help="If enabled, Uchko selects the next skill and difficulty automatically.",
        )

    with cB:
        if not bool(st.session_state.adaptive_mode):
            st.selectbox(
                "Manual skill",
                skill_ids,
                index=skill_ids.index(st.session_state.manual_skill_id) if st.session_state.manual_skill_id in skill_ids else 0,
                format_func=_format_skill_label,
                key="manual_skill_id",
            )
        else:
            st.caption("Manual skill hidden (Adaptive mode ON).")

    with cC:
        if not bool(st.session_state.adaptive_mode):
            st.selectbox("Manual difficulty", [1, 2, 3], key="manual_difficulty")
        else:
            st.caption("Manual difficulty hidden (Adaptive mode ON).")

    adaptive_mode = bool(st.session_state.adaptive_mode)

    selected_skill_name = skills[chosen_skill_id].name if chosen_skill_id in skills else chosen_skill_id
    st.info(
        f"**Selected:** {selected_skill_name} (difficulty {chosen_difficulty})\n\n"
        f"{decision.reason if adaptive_mode else 'Manual selection.'}"
    )

    interv = decision.intervention
    if interv.message:
        st.warning(f"**Intervention:** {interv.message}")
    if interv.recommend_review_skill_id:
        rsid = interv.recommend_review_skill_id
        st.info(f"**Recommended review:** {skills[rsid].name} ({rsid})")

    with st.expander("Decision details", expanded=False):
        st.write(f"- Risk source: **{risk_source}**")
        st.write(f"- Risk raw: **{float(risk_raw):.3f}**")
        st.write(f"- Risk used: **{'warming up' if not risk_ready else f'{float(risk_score_used):.3f}'}**")
        if guard_reason:
            st.warning(guard_reason)

    st.divider()

    # Question generation
    def _new_question(skill_id: str, difficulty: int) -> None:
        enable_llm_now = bool(st.session_state.use_llm) and (st.session_state.llm_calls_used < int(st.session_state.max_llm_calls))
        q = generate_question(
            templates,
            skill_id=skill_id,
            difficulty=int(difficulty),
            enable_llm=enable_llm_now,
            skill_name=skills[skill_id].name if skill_id in skills else skill_id,
        )
        st.session_state.current_q = q
        st.session_state.q_start_ts = time.time()
        st.session_state.attempts.setdefault(q.question_id, 0)

        used = bool(getattr(q, "generated_params", {}).get("_llm_used", False))
        st.session_state.llm_used = used
        if used:
            st.session_state.llm_calls_used += 1

    colX, colY, colZ = st.columns([1, 1, 2])
    with colX:
        if st.button("Generate question", key="btn_gen_q"):
            _new_question(chosen_skill_id, chosen_difficulty)
            st.rerun()
    with colY:
        if st.button("End session", key="btn_end_session"):
            append_event(make_end_event(student_id, session_id), events_path=EVENTS_PATH)
            st.success("Session ended (event logged).")
            try:
                df_all_u = load_events(EVENTS_PATH, student_id=student_id)
                df_sess = df_all_u[df_all_u["session_id"] == session_id].copy()

                row = summarize_session(
                    repo_root=ROOT,
                    skills_path=SKILLS_PATH,
                    df_session=df_sess,
                    student_id=student_id,
                    session_id=session_id,
                    goal_skill_id=st.session_state.session_goal_skill_id,
                )
                upsert_session_summary(out_path=SESSION_SUMMARIES_PATH, row=row)
                st.info("Saved session summary.")
            except Exception as e:
                st.warning(f"Could not save session summary: {e}")

    with colZ:
        st.caption("Tip: Keep Adaptive mode on, set a goal, then generate questions repeatedly.")

    q = st.session_state.get("current_q")
    if not q:
        st.warning("Click **Generate question** to begin.")
        st.stop()

    st.markdown(
        f"""
<div class="card card-big">
  <div class="card-title">Question</div>
  <div style="font-size:1.1rem; line-height:1.55;">{q.prompt}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    if bool(st.session_state.llm_used):
        st.caption("✨ Enhanced by Gemini")
    else:
        st.caption("🧩 Template question (stable demo mode)")

    # Hint / explanation
    h1, h2 = st.columns(2)
    with h1:
        if st.button("Hint", key="btn_hint"):
            # Pick a coaching hint for the current question's skill
            hints = get_skill_hints(q.skill_id, n=3)
            hint_text = random.choice(hints) if hints else "Try breaking the problem into smaller steps."

            append_event(
                make_hint_event(
                    student_id=student_id,
                    session_id=session_id,
                    skill_id=q.skill_id,
                    question_id=q.question_id,
                    difficulty=q.difficulty,
                    meta={"hint": hint_text},
                ),
                events_path=EVENTS_PATH,
            )

            st.info(f"Hint: {hint_text}")
    with h2:
        if st.button("Explanation", key="btn_expl"):
            exps = get_skill_explanations(q.skill_id, n=3)
            exp_text = random.choice(exps) if exps else "Think about the rule being applied and work step-by-step."

            append_event(
                make_explanation_event(
                    student_id=student_id,
                    session_id=session_id,
                    skill_id=q.skill_id,
                    question_id=q.question_id,
                    difficulty=q.difficulty,
                    meta={"explanation": exp_text},
                ),
                events_path=EVENTS_PATH,
            )

            st.info(f"Explanation: {exp_text}")

    # Solve logging
    def _log_solve(is_correct: int) -> None:
        st.session_state.attempts[q.question_id] = st.session_state.attempts.get(q.question_id, 0) + 1
        attempt_number = st.session_state.attempts[q.question_id]

        start_ts = st.session_state.q_start_ts or time.time()
        rt_ms = int((time.time() - start_ts) * 1000)

        append_event(
            make_solve_event(
                student_id=student_id,
                session_id=session_id,
                skill_id=q.skill_id,
                question_id=q.question_id,
                difficulty=q.difficulty,
                correct=int(is_correct),
                response_time_ms=int(rt_ms),
                attempt_number=int(attempt_number),
                meta={"template_id": q.template_id, "generated_params": q.generated_params},
            ),
            events_path=EVENTS_PATH,
        )

    st.write("")
    if q.type == "mcq" and q.choices:
        ans = st.radio("Choose an answer", q.choices, key=f"ans_{q.question_id}")
        if st.button("Submit", key="btn_submit_mcq"):
            is_correct = int(ans == q.correct_answer)
            _log_solve(is_correct)

            if is_correct:
                st.success("✅ Correct!")
            else:
                st.error(f"❌ Incorrect. Correct answer: **{q.correct_answer}**")

            _new_question(chosen_skill_id, chosen_difficulty)
            st.rerun()
    else:
        ans = st.text_input("Your answer", key=f"ans_{q.question_id}")
        if st.button("Submit", key="btn_submit_text"):
            is_correct = int(str(ans).strip() == str(q.correct_answer).strip())
            _log_solve(is_correct)

            if is_correct:
                st.success("✅ Correct!")
            else:
                st.error(f"❌ Incorrect. Correct answer: **{q.correct_answer}**")

            _new_question(chosen_skill_id, chosen_difficulty)
            st.rerun()


# Progress tab
with tab_progress:
    st.subheader("Progress")

    rows = []
    for sid in skill_ids:
        prereqs = getattr(skills[sid], "prerequisites", []) or []
        rows.append(
            {
                "skill_id": sid,
                "skill": skills[sid].name,
                "mastery": float(mastery_map.get(sid, 0.0)),
                "prerequisites": ", ".join(prereqs),
            }
        )
    df_mastery = pd.DataFrame(rows).sort_values("mastery", ascending=True)

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.dataframe(df_mastery, use_container_width=True, height=520)

    with c2:
        st.markdown('<div class="card card-tight"><div class="card-title">Session stats</div></div>', unsafe_allow_html=True)
        st.metric("Events", metrics["n_events"])
        st.metric("Solves", metrics["n_solves"])
        st.metric("Accuracy", f"{metrics['acc']:.2f}")
        if metrics["mean_rt_ms"] > 0:
            st.metric("Avg response time", f"{metrics['mean_rt_ms']/1000:.1f}s")
        st.metric("Hints", metrics["hints"])
        st.metric("Explanations", metrics["expl"])

        st.write("")
        st.markdown('<div class="card card-tight"><div class="card-title">Risk</div></div>', unsafe_allow_html=True)
        if not risk_ready:
            st.info(f"Warming up (need {MIN_SOLVES_FOR_RISK} solves). Current: {n_solves}")
        else:
            st.metric("Risk (smoothed)", f"{risk_score_used:.2f}")
            st.caption(f"Level: {risk_cat.upper()} · Source: {risk_source}")
            if guard_reason:
                st.warning(guard_reason)


# CURRICULUM TAB
with tab_curriculum:
    st.subheader("Curriculum")

    st.markdown(
        '<div class="card"><div class="card-title">Curriculum map</div><div class="muted">Prerequisites + mastery overlay.</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if draw_curriculum_graph is None:
        st.info("Graph renderer not available (missing dependency). You can still browse prerequisites below.")
    else:
        try:
            fig = draw_curriculum_graph(
                skills=skills,
                mastery=mastery_map,
                goal_skill_id=goal_skill_id,
                recommended_skill_id=chosen_skill_id,
                figsize=(12, 7),
            )
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.warning("Could not render curriculum graph.")
            st.code(str(e))

    st.divider()
    st.subheader("Prerequisites browser")

    sid = st.selectbox("Select skill", skill_ids, format_func=_format_skill_label, key="curriculum_skill_pick")
    prereqs = getattr(skills[sid], "prerequisites", []) or []
    st.write(f"**Prerequisites for {skills[sid].name}:**")
    if prereqs:
        for p in prereqs:
            name = skills[p].name if p in skills else p
            st.write(f"- {name} ({p}) — mastery {mastery_map.get(p, 0.0):.2f}")
    else:
        st.caption("No prerequisites for this skill.")


# SETTINGS TAB
with tab_settings:
    st.subheader("Settings / Debug")
    st.subheader("User session history")

    if SESSION_SUMMARIES_PATH.exists():
        df_hist = pd.read_parquet(SESSION_SUMMARIES_PATH)

        # filter sessions for selected user
        df_user = df_hist[df_hist["student_id"] == student_id].copy()

        if df_user.empty:
            st.info("No previous sessions for this user.")
        else:
            # newest sessions first
            df_user = df_user.sort_values("session_end_ts", ascending=False)

            st.dataframe(
                df_user[
                    [
                    "session_id",
                    "session_end_ts",
                    "n_solves",
                    "acc",
                    "risk_raw",
                    "risk_level",
                    "mastery_goal",
                    ]
                ],
                use_container_width=True,
            )
            st.divider()
            st.subheader("What to work on next")

            latest = df_user.iloc[0]

            weak = json.loads(latest.get("weak_skills_json", "[]"))
            strong = json.loads(latest.get("strong_skills_json", "[]"))

            if weak:
                for sid, p in weak:
                    name = skills[sid].name if sid in skills else sid
                    st.write(f"- **{name}** ({sid}) — mastery {float(p):.2f}")
            else:
                st.caption("No weak skill summary available yet.")

            st.subheader("Strong skills")

            if strong:
                for sid, p in strong:
                    name = skills[sid].name if sid in skills else sid
                    st.write(f"- **{name}** ({sid}) — mastery {float(p):.2f}")
            else:
                st.caption("No strong skill summary available yet.")
    else:
        st.info("Session history file not found yet.")


    st.markdown(
        '<div class="card"><div class="card-title">Preferences</div><div class="muted">This page avoids duplicate widget keys. Adaptive mode is controlled in the Practice tab only.</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")

    st.write(f"Adaptive mode is currently: **{bool(st.session_state.adaptive_mode)}**")
    st.write(f"Manual skill: **{_format_skill_label(st.session_state.manual_skill_id)}**")
    st.write(f"Manual difficulty: **{int(st.session_state.manual_difficulty)}**")

    st.divider()

    with st.expander("Risk details & model status", expanded=False):
        st.write(f"Risk source: **{risk_source}**")
        st.write(f"Risk raw: **{float(risk_raw):.3f}**")
        st.write(f"Risk used: **{'warming up' if not risk_ready else f'{float(risk_score_used):.3f}'}**")
        if guard_reason:
            st.warning(guard_reason)

        model_obj, feat_order, _thr = try_load_model(ROOT)
        st.write(f"Model loaded: **{model_obj is not None}**")
        if model_obj is not None:
            st.write(f"Model type: `{type(model_obj)}`")
            st.write(f"#features: **{len(feat_order)}**")

        st.write("Risk features:")
        st.json(asdict(rf))

    with st.expander("LLM debug (optional)", expanded=False):
        st.caption("LLM is OFF by default for demo stability.")

        use_llm_setting = st.toggle("Use Gemini enhancements", value=bool(st.session_state.use_llm), key="use_llm_setting")
        max_llm_calls_setting = st.slider(
            "Max Gemini calls this session", 0, 10, int(st.session_state.max_llm_calls), key="max_llm_calls_setting"
        )

        #updating non-widget state
        st.session_state.use_llm = bool(use_llm_setting)
        st.session_state.max_llm_calls = int(max_llm_calls_setting)

        st.write("UCHKO_LLM_ENABLED:", os.getenv("UCHKO_LLM_ENABLED"))
        st.write("UCHKO_LLM_PROVIDER:", os.getenv("UCHKO_LLM_PROVIDER"))
        st.write("UCHKO_LLM_MODEL:", os.getenv("UCHKO_LLM_MODEL"))
        st.write("GEMINI_API_KEY set:", bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")))
        st.write("Last LLM status:", getattr(llm_enh, "LAST_LLM_STATUS", "unknown") if llm_enh else "llm module not available")
        st.write(f"Calls used: {st.session_state.llm_calls_used}/{st.session_state.max_llm_calls}")

    with st.expander("Event logs (session)", expanded=False):
        st.dataframe(df_session, use_container_width=True)

    with st.expander("Event logs (all-time)", expanded=False):
        st.dataframe(df_all, use_container_width=True)
