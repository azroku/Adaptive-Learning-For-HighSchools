# Uchko — Adaptive Learning Platform (Demo)

Uchko is an **adaptive learning platform prototype** for high school mathematics.  
It demonstrates how student interaction data can be used to power **personalized practice**, **knowledge tracing**, and **risk-aware instructional decisions** in real time.

The project was developed as an **academic demo**, but its architecture is intentionally designed to scale into a full production system.

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Models Used](#models-used)
- [Installation](#installation)
- [Usage](#usage)
- [Risk Model Training (Optional)](#risk-model-training-optional)
- [Design Decisions](#design-decisions)
- [Current Limitations](#current-limitations)
- [Planned Improvements](#planned-improvements)

---

## Project Overview

### Purpose

Uchko demonstrates a **complete adaptive learning loop**:

1. Generate curriculum-aligned mathematics questions  
2. Log student interactions as structured events  
3. Estimate skill mastery using knowledge tracing  
4. Estimate learning risk using behavioral signals  
5. Adapt future practice in real time  

### Problem Addressed

Traditional learning platforms often:

- Treat all learners identically  
- Detect learning difficulties too late  
- Hide decision logic inside opaque systems  

Uchko addresses these issues by being:

- **Event-driven and transparent**
- **Modular and inspectable**
- **Safe for experimentation**, with explicit guards and fallbacks

---

## System Architecture

The platform is composed of **loosely coupled, inspectable modules**:

```text
uchko/
├── app/
│   └── streamlit_app.py        # Main Streamlit application
│
├── uchko_core/
│   ├── content/               # Skills, templates, hints, explanations
│   ├── events.py              # Event schema and factories
│   ├── event_store.py         # Event persistence (Parquet)
│   ├── kt/                    # Knowledge tracing (BKT)
│   ├── risk/                  # Risk features, ML scoring, guards
│   ├── adaptive/              # Risk-aware adaptive policy
│   ├── users.py               # Lightweight user accounts
│   ├── analytics/             # Session summary
│   ├── llm/                   # Inclusion of Large Language Models (LLM)
│   └── viz/                   # Curriculum graph visualization
│
├── scripts/
│   ├── train_risk_model_edm_gbm.py
│   ├── pregen_llm_cache.py
│   └── fit_bkt_params.py
│
├── tests/
│   └── test_generator.py
│
├── data/
│   ├── content/               # skills.json, templates.json
│   ├── cache/                 # events.parquet, users.json, session history
│   ├── prod_event_logs.csv    # user data
│   └── edm_cup_2023/          # (training only)
│
├── models/
│   ├── bkt_params.json
│   └── edm_risk_gbm/
│       ├── risk_model.joblib
│       ├── feature_spec.json
│       └── training_metrics.json
│
├── requirements.txt
└── README.md
```
## Core Components

### Question Generator
Template-based mathematics question generation aligned to **skills** and **difficulty levels**.

---

### Event Logging
Every learner interaction is logged as a structured event:

- `start`
- `solve`
- `hint`
- `explanation`
- `end`

All events are persisted in **Parquet format** for efficiency, scalability, and auditability.

---

### Knowledge Tracing
Uses **Bayesian Knowledge Tracing (BKT)** to estimate mastery for each skill.

- Mastery is recomputed from events to avoid session-state drift  
- BKT parameters can be fitted using real student data  

---

### Risk Scoring
Learning risk is estimated using a **machine-learning model trained on EDM Cup 2023 data**.

- Behavioral, session-level features  
- Probabilistic output (`P(at-risk)`)  
- Threshold applied at inference time  
- Explicit guards for cold start and extreme cases  

---

### Adaptive Policy
Selects the next skill and difficulty based on:

- Estimated mastery  
- Estimated risk  
- Session goal  

---

### Session Goals & History
- Users can set explicit mastery goals  
- Each session is summarized and stored in Parquet  
- Full session history can be inspected per user  

---

## Models Used

### Risk Detection Model

- **Model**: HistGradientBoostingClassifier  
- **Library**: scikit-learn  
- **Training Data**: EDM Cup 2023  

**Features (8):**
- `n_solves`
- `acc`
- `recent_acc_10`
- `mean_rt_ms`
- `p90_rt_ms`
- `wrong_streak_max`
- `hints_per_solve`
- `explanations_per_solve`

**Final Performance:**
- ROC-AUC: **0.725**
- Average Precision: **0.777**
- Accuracy: **0.67**

**Threshold Policy:**
- `risk ≥ 0.65` → at-risk  
- `risk ≥ 0.80` → high risk  

Thresholds are chosen to prioritize **precision** (fewer false positives).

---

### Knowledge Tracing Model

- **Model**: Bayesian Knowledge Tracing (BKT)

**Parameters:**
- `p_init`
- `p_transit`
- `p_guess`
- `p_slip`

Parameters can be fitted via **maximum likelihood** on logged student events.

---

## Installation

### Requirements
- **Python 3.12** (recommended)

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```
## Typical User Flow

1. Create or select a user  
2. Start a learning session  
3. Set a goal skill (optional)  
4. Practice questions  
5. Request hints or explanations  
6. Observe mastery and risk updates  
7. Review session history  

---

## Risk Model Training (Optional)

The application **does not require EDM data to run**.  
EDM Cup 2023 data is only needed to **retrain the risk model**.

### Required EDM Files

- `action_logs.csv`
- `training_unit_test_scores.csv`
- `assignment_relationships.csv`

### Training Command

```bash
python scripts/train_risk_model_edm_gbm.py \
  --data_dir data/edm_cup_2023 \
  --out_dir models/edm_risk_gbm
```
## Design Decisions

### Event-based State
All system state is derived from logged events.

### Probability-first Modeling
Models output probabilities; product logic applies thresholds.

### Parquet over CSV
Chosen for performance, schema stability, and scalability.

### Explicit Model Guards
Prevent implausible predictions during cold start.

### No Mandatory LLM Dependency
The system is fully functional offline.

---

## Current Limitations

- Small, manually defined skill ontology  
- Risk model trained on external dataset  
- UI prioritizes clarity over polish  

---

## Planned Improvements

- Risk-aware difficulty pacing  
- Early-warning risk detection  
- Skill-specific risk models  
- Advanced mastery models (DKT, IRT)  
- Teacher dashboards and cohort analytics  

---


This project is currently an **academic prototype**.

