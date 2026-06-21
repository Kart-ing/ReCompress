from rezero.checkpoint import CheckpointBuilder, CheckpointStack
from engine.tokens import count_tokens

def test_checkpoint_under_cap():
    builder = CheckpointBuilder(goal="Find the founder", use_llm=False)
    history = [
        {"role": "user",      "content": "Tell me about Alice and Tech Corp"},
        {"role": "assistant", "content": "Alice founded Tech Corp in 2010 and serves as CEO"},
    ]
    cp = builder.build(history, trauma="Alice Tech Corp")
    assert count_tokens(cp) <= 150

def test_stack_push_and_current():
    stack = CheckpointStack()
    stack.push("state one", turn=1)
    stack.push("state two", turn=2)
    assert stack.current() == "state two"

def test_stack_revert():
    stack = CheckpointStack()
    e1 = stack.push("state one",   turn=1)
    _  = stack.push("state two",   turn=2)
    _  = stack.push("state three", turn=3)
    reverted = stack.revert_to(e1.id)
    assert reverted == "state one"
    assert stack.current() == "state one"
    assert len(stack.stack) == 1

def test_stack_capped_at_10():
    stack = CheckpointStack()
    for i in range(15):
        stack.push(f"state {i}", turn=i)
    assert len(stack.stack) <= 10

def test_stack_revert_unknown_id_raises():
    stack = CheckpointStack()
    stack.push("only state", turn=1)
    try:
        stack.revert_to(999)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_session_revert_preserves_trauma():
    from rezero.session import ReZeroSession
    s = ReZeroSession(goal="Find Alice the founder")
    s.add_turn("Alice founded Tech Corp", "Correct, Alice is the founder")
    s.add_turn("What did Bob do?", "Bob joined later")
    trauma_before = s.trauma_extractor.get()
    ids = s.list_checkpoints()
    if ids:
        s.revert_to(ids[0])
    assert s.trauma_extractor.get() == trauma_before
