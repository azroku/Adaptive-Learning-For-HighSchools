from pathlib import Path

from uchko_core.content.load import load_templates
from uchko_core.content.generator import generate_question

ROOT = Path(__file__).resolve().parents[1]

def test_generate_questions_smoke():
    templates = load_templates(ROOT / "data" / "content" / "templates.json")

    # generate a few known skills
    for skill_id in ["S01_ARITH", "S05_LIN_EQ", "S12_STATS"]:
        q = generate_question(templates, skill_id=skill_id, difficulty=1, seed=123)
        assert q.skill_id == skill_id
        assert isinstance(q.prompt, str) and len(q.prompt) > 0
        assert isinstance(q.correct_answer, str) and len(q.correct_answer) > 0

        if q.type == "mcq":
            assert q.choices is not None
            assert len(q.choices) == 4
            assert q.correct_answer in q.choices
