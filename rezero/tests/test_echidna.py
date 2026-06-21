from rezero.echidna import Echidna

def test_mock_pass_on_short_history():
    e = Echidna(use_llm=False)
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    d = e.decide(history, "Alice", "no checkpoint", 1, [])
    assert d.action == "pass"

def test_mock_checkpoint_on_token_threshold():
    e = Echidna(use_llm=False)
    history = [{"role": "user", "content": "word " * 900}]
    d = e.decide(history, "Alice", "cp", 1, [])
    assert d.action == "checkpoint"
    assert d.urgency == "high"

def test_mock_checkpoint_on_cadence():
    e = Echidna(use_llm=False)
    history = [{"role": "user", "content": "short"}, {"role": "assistant", "content": "reply"}]
    d = e.decide(history, "Alice", "cp", turns_since_checkpoint=5, available_checkpoints=[])
    assert d.action == "checkpoint"
    assert d.urgency == "medium"

def test_decision_fields_valid():
    e = Echidna(use_llm=False)
    d = e.decide([], "", "", 0, [])
    assert d.action  in ("checkpoint", "revert", "pass")
    assert d.urgency in ("low", "medium", "high")
    assert isinstance(d.reason, str)

def test_echidna_wired_into_session():
    from rezero.session import ReZeroSession
    s = ReZeroSession(goal="Test Alice at Tech Corp", use_llm=False)
    for i in range(6):
        s.add_turn(f"Question {i} about Tech Corp", f"Answer {i} about Tech Corp")
    assert len(s.checkpoint_stack.list_ids()) >= 1

def test_get_checkpoint_is_pure_read():
    from rezero.session import ReZeroSession
    s = ReZeroSession(goal="Test", use_llm=False)
    s.add_turn("hello", "hi")
    before = s.checkpoint_stack.list_ids()[:]
    _ = s._get_checkpoint()
    assert s.checkpoint_stack.list_ids() == before
