from docuquery import DocuQuery


def make():
    dq = DocuQuery(size=40, overlap=10)
    dq.add_directory("data")
    return dq


def test_retrieves_right_document():
    dq = make()
    assert "batteries" in dq.search("how many charge cycles does lithium iron phosphate last")[0][0].source
    assert "heatpumps" in dq.search("coefficient of performance")[0][0].source


def test_answer_has_citation_and_text():
    r = make().ask("How much does solar panel output drop each year?")
    assert "0.5 percent" in r["answer"]
    assert r["sources"][0]["source"].endswith("solar.md")


def test_unknown_question():
    assert make().ask("zzzz qqqq")["sources"] == []


def test_custom_generator():
    r = make().ask("what is a heat pump", generator=lambda q, ch: f"{len(ch)} chunks")
    assert r["answer"].endswith("chunks")
