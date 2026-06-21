from baselines.naive import NaiveSession

LONG_USER = "This is a detailed question about Tech Corp funding rounds and investor relations in Silicon Valley"
LONG_ASST = "Tech Corp has raised several hundred million dollars across multiple rounds from top-tier investors"
N_TURNS = 20

def test_naive_grows_with_turns():
    n = NaiveSession(goal="Research Tech Corp")
    counts = []
    for i in range(N_TURNS):
        n.add_turn(f"Turn {i}: {LONG_USER}", f"Turn {i}: {LONG_ASST}")
        counts.append(n.token_count())
    assert counts[-1] > counts[0]
    assert counts[-1] > counts[0] * 1.5, \
        f"Naive grew too slowly: first={counts[0]}, last={counts[-1]}"

def test_naive_vs_rbd_after_many_turns():
    from rezero.session import ReZeroSession
    rbd   = ReZeroSession(goal="Research Tech Corp funding")
    naive = NaiveSession(goal="Research Tech Corp funding")
    for i in range(N_TURNS):
        rbd.add_turn(f"Turn {i}: {LONG_USER}", f"Turn {i}: {LONG_ASST}")
        naive.add_turn(f"Turn {i}: {LONG_USER}", f"Turn {i}: {LONG_ASST}")
    assert naive.token_count() > rbd.token_count(), \
        f"Naive ({naive.token_count()}) should exceed RbD ({rbd.token_count()}) after {N_TURNS} turns"

def test_naive_has_full_history_in_prompt():
    n = NaiveSession(goal="Test")
    n.add_turn("First message about Alice", "First reply about Alice founding Tech Corp")
    n.add_turn("Second message about Bob",  "Second reply about Bob joining later")
    prompt = n.prompt_for_solver()
    assert "First message"  in prompt
    assert "Second message" in prompt

def test_naive_interface_matches_rbd():
    from rezero.session import ReZeroSession
    rbd   = ReZeroSession(goal="test")
    naive = NaiveSession(goal="test")
    rbd.add_turn("q", "a")
    naive.add_turn("q", "a")
    assert isinstance(rbd.prompt_for_solver(),   str)
    assert isinstance(naive.prompt_for_solver(), str)
    assert isinstance(rbd.token_count(),         int)
    assert isinstance(naive.token_count(),       int)

def test_cumulative_naive_exceeds_rbd_2_5x():
    from rezero.session import ReZeroSession
    rbd   = ReZeroSession(goal="Research Tech Corp funding")
    naive = NaiveSession(goal="Research Tech Corp funding")
    cum_rbd, cum_naive = 0, 0
    for i in range(N_TURNS):
        rbd.add_turn(f"Turn {i}: {LONG_USER}", f"Turn {i}: {LONG_ASST}")
        naive.add_turn(f"Turn {i}: {LONG_USER}", f"Turn {i}: {LONG_ASST}")
        cum_rbd   += rbd.token_count()
        cum_naive += naive.token_count()
    assert cum_naive > cum_rbd * 2.5, \
        f"Cumulative naive ({cum_naive}) should be >2.5x RbD ({cum_rbd}) after {N_TURNS} turns"
