import json
from pathlib import Path
from rezero.session import ReZeroSession
from engine.tokens import count_tokens

def _load_convo():
    candidates = [
        Path("demo/scripted_convo.jsonl"),
        Path("../demo/scripted_convo.jsonl"),
        Path(__file__).parent.parent / "demo" / "scripted_convo.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    raise FileNotFoundError("scripted_convo.jsonl not found")

def test_token_count_never_exceeds_300():
    convo = _load_convo()
    s = ReZeroSession(goal=convo[0].get("goal", "Research task"), use_llm=False)
    counts = []
    for turn in convo:
        s.add_turn(turn["user"], turn["assistant"])
        counts.append(s.token_count())
    assert all(c <= 300 for c in counts), f"Exceeded 300: {list(enumerate(counts))}"

def test_token_count_is_flat():
    convo = _load_convo()
    s = ReZeroSession(goal=convo[0].get("goal", "Research task"), use_llm=False)
    counts = []
    for turn in convo:
        s.add_turn(turn["user"], turn["assistant"])
        counts.append(s.token_count())
    n = len(counts)
    quarter = max(1, n // 4)
    early_avg = sum(counts[:quarter]) / quarter
    late_avg  = sum(counts[-quarter:]) / quarter
    assert abs(late_avg - early_avg) < 80, \
        f"Not flat — early: {early_avg:.0f}, late: {late_avg:.0f}"

def test_trauma_section_always_present():
    convo = _load_convo()
    s = ReZeroSession(goal=convo[0].get("goal", "Research task"), use_llm=False)
    for turn in convo:
        s.add_turn(turn["user"], turn["assistant"])
    prompt = s.prompt_for_solver()
    trauma_section = prompt.split("[CHECKPOINT]")[0].replace("[TRAUMA]", "").strip()
    assert len(trauma_section) > 0

def test_all_three_sections_present_every_turn():
    convo = _load_convo()
    s = ReZeroSession(goal=convo[0].get("goal", "Research task"), use_llm=False)
    for turn in convo:
        s.add_turn(turn["user"], turn["assistant"])
        prompt = s.prompt_for_solver()
        assert "[TRAUMA]"     in prompt
        assert "[CHECKPOINT]" in prompt
        assert "[DELTA]"      in prompt

def test_delta_always_contains_latest_turn():
    convo = _load_convo()
    s = ReZeroSession(goal=convo[0].get("goal", "Research task"), use_llm=False)
    for turn in convo[:-1]:
        s.add_turn(turn["user"], turn["assistant"])
    last = convo[-1]
    s.add_turn(last["user"], last["assistant"])
    prompt = s.prompt_for_solver()
    delta_section = prompt.split("[DELTA]")[1].strip()
    last_words = last["user"].split()[:3]
    assert any(w in delta_section for w in last_words)

def test_context_builder_owns_assembly():
    s = ReZeroSession(goal="Test", use_llm=False)
    assert not hasattr(s, "_assemble"), \
        "_assemble should be in ContextBuilder, not session"
