from rezero.session import ReZeroSession

def test_prompt_has_three_sections():
    s = ReZeroSession(goal="Test goal")
    s.add_turn("Hello", "Hi there")
    prompt = s.prompt_for_solver()
    assert "[TRAUMA]"     in prompt
    assert "[CHECKPOINT]" in prompt
    assert "[DELTA]"      in prompt

def test_token_count_under_300():
    s = ReZeroSession(goal="Find the founder of the company that NASA works with most")
    for i in range(10):
        s.add_turn(
            f"User message number {i} about SpaceX and NASA contracts with extra context",
            f"Assistant reply {i} explaining the relationship between SpaceX and government"
        )
    assert s.token_count() <= 300

def test_delta_is_most_recent_user_turn():
    s = ReZeroSession(goal="Test")
    s.add_turn("First question here", "First answer")
    s.add_turn("Second question here", "Second answer")
    prompt = s.prompt_for_solver()
    delta_section = prompt.split("[DELTA]")[1]
    assert "Second question" in delta_section

def test_goal_seeded_into_trauma():
    s = ReZeroSession(goal="Find Alice the Chief Executive of Tech Corp")
    prompt = s.prompt_for_solver()
    trauma_section = prompt.split("[CHECKPOINT]")[0].replace("[TRAUMA]", "").strip()
    assert "Alice" in trauma_section or "Tech" in trauma_section

def test_token_flat_over_many_turns():
    N_TURNS = 25
    s = ReZeroSession(goal="Research Alice and Tech Corp funding history")
    counts = []
    for i in range(N_TURNS):
        s.add_turn(
            f"Question {i} about Tech Corp funding rounds in Silicon Valley this year",
            f"Answer {i}: Tech Corp raised several million dollars from investors recently"
        )
        counts.append(s.token_count())
    assert all(c <= 300 for c in counts), f"Exceeded 300 at some turn: {list(enumerate(counts))}"
