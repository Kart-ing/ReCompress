from rezero.trauma import TraumaExtractor

def test_mock_extracts_entities():
    t = TraumaExtractor(use_llm=False)
    t.update("Alice founded Tech Corp in 2010")
    result = t.get()
    assert "Alice" in result or "Tech" in result

def test_cap_enforced():
    t = TraumaExtractor(use_llm=False)
    for i in range(20):
        t.update(f"Entity {i} is a named person from Some Place {i} and works at Big Company {i}")
    assert len(t.get().split()) <= 55

def test_trauma_accumulates_specific_entity():
    t = TraumaExtractor(use_llm=False)
    t.update("Alice is the founder")
    t.update("Bob is the Chief Executive of Widget Corp")
    result = t.get()
    assert "Alice" in result or "Bob" in result or "Widget" in result

def test_empty_message():
    t = TraumaExtractor(use_llm=False)
    t.update("")
    assert t.get() == ""

def test_no_entities_no_change():
    t = TraumaExtractor(use_llm=False)
    t.update("alice founded techcorp")
    assert t.get() == ""

def test_second_update_adds_new_entity():
    t = TraumaExtractor(use_llm=False)
    t.update("Alice founded the company")
    assert "Alice" in t.get()
    t.update("Widget Corp is the company name")
    assert "Widget" in t.get()
