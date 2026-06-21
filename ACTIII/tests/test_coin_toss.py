from baselines.coin_toss import coin_toss_solve
from engine.tokens import count_tokens
from unittest.mock import patch

MOCK_SOLVE_RETURN = {"answer": "test", "prompt_tokens": 100}
SAMPLE_CONTEXT = "Alice founded Tech Corp in 2010. She raised fifty million dollars from Sequoia Capital. Bob joined as CTO in 2012. The company went public in 2015 with a valuation of two billion dollars."

def test_rate_zero_keeps_all():
    with patch("baselines.coin_toss.solve", return_value=MOCK_SOLVE_RETURN):
        result = coin_toss_solve(SAMPLE_CONTEXT, "Who founded Tech Corp?", removal_rate=0.0)
        assert count_tokens(SAMPLE_CONTEXT) > 20
        assert result["post_compression_tokens"] == count_tokens(SAMPLE_CONTEXT)

def test_rate_one_removes_almost_all():
    with patch("baselines.coin_toss.solve", return_value=MOCK_SOLVE_RETURN):
        result = coin_toss_solve(SAMPLE_CONTEXT, "Who founded Tech Corp?", removal_rate=1.0)
        assert result["post_compression_tokens"] < count_tokens(SAMPLE_CONTEXT) * 0.2

def test_rate_half_roughly_halves():
    with patch("baselines.coin_toss.solve", return_value=MOCK_SOLVE_RETURN):
        full_tokens = count_tokens(SAMPLE_CONTEXT)
        result = coin_toss_solve(SAMPLE_CONTEXT, "Who?", removal_rate=0.5)
        ratio = result["post_compression_tokens"] / full_tokens
        assert 0.25 < ratio < 0.75

def test_reproducible_same_seed():
    with patch("baselines.coin_toss.solve", return_value=MOCK_SOLVE_RETURN):
        r1 = coin_toss_solve(SAMPLE_CONTEXT, "Q?", removal_rate=0.5, seed=42)
        r2 = coin_toss_solve(SAMPLE_CONTEXT, "Q?", removal_rate=0.5, seed=42)
        assert r1["post_compression_tokens"] == r2["post_compression_tokens"]

def test_different_seeds_differ():
    with patch("baselines.coin_toss.solve", return_value=MOCK_SOLVE_RETURN):
        r1 = coin_toss_solve(SAMPLE_CONTEXT, "Q?", removal_rate=0.5, seed=42)
        r2 = coin_toss_solve(SAMPLE_CONTEXT, "Q?", removal_rate=0.5, seed=99)
        assert r1["post_compression_tokens"] != r2["post_compression_tokens"]

def test_question_never_modified():
    question = "Who founded Tech Corp?"
    with patch("baselines.coin_toss.solve", return_value=MOCK_SOLVE_RETURN) as mock:
        coin_toss_solve(SAMPLE_CONTEXT, question, removal_rate=0.9)
        call_args = mock.call_args
        assert call_args[0][1] == question or call_args[1].get("question") == question

def test_returns_all_fields():
    with patch("baselines.coin_toss.solve", return_value=MOCK_SOLVE_RETURN):
        result = coin_toss_solve(SAMPLE_CONTEXT, "Q?", removal_rate=0.5)
        assert "answer" in result
        assert "prompt_tokens" in result
        assert "pre_compression_tokens" in result
        assert "post_compression_tokens" in result
