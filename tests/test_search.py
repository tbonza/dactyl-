from spondee.search import (
    identify_statements,
    sentence_metadata,
    search_text,
)

import pytest


@pytest.fixture
def example_text():
    txt = "".join(
        [
            "Gavin Sheets hit his first career grand slam, and the ",
            "Chicago White Sox won their second straight after a ",
            "franchise-record 14-game losing streak, beating the ",
            "Boston Red Sox 6-1 on Saturday.",
        ]
    )
    return txt


def valid_output(res):
    assert len(res) == 2

    print(res)

    assert res[0].subject == ["Gavin Sheets"]
    assert res[0].predicate == ["first career grand slam"]

    assert res[1].subject == ["Chicago White Sox"]
    assert res[1].predicate[-2] == "Boston Red Sox"


def test_sentence_metadata(load_nlp_pipeline, example_text):
    docs = load_nlp_pipeline(example_text)

    example = docs.sentences[0]
    tree = example.constituency
    meta = example.to_dict()

    statements = identify_statements(tree)

    got = sentence_metadata(0, statements, meta)
    assert len(got) == 2

    assert got[0].subject[0].text == "Gavin"
    assert got[0].subject[1].text == "Sheets"


def test_search_text(load_nlp_pipeline, example_text):
    res = search_text(example_text, load_nlp_pipeline)
    valid_output(res)


@pytest.fixture
def error_text0():
    txt = "".join(
        [
            "Boston’s Bobby Dalbec homered leading off the fifth. ",
            "But manager Alex Cora got ejected by plate umpire Alan Porter ",
            "after pinch-hitter Jamie Westbrook struck out looking at a pitch ",
            "in the lower part of the zone for the third out of the inning.",
        ]
    )
    return txt


def test_error0_search_text(load_nlp_pipeline, error_text0):
    res = search_text(error_text0, load_nlp_pipeline)

    assert len(res) == 4
    print(res)
    assert res[1].sidx == 0
    assert res[1].subject == ["Bobby Dalbec"]
    assert "homered" in res[1].predicate_text


def test_error_text1(load_nlp_pipeline):
    """Should have two sentences not one."""
    txt = "".join(
        [
            "The states — led by West Virginia, Georgia, Iowa and North Dakota ",
            "— said in the lawsuit that the so-called Waters of the United ",
            "States (WOTUS) rule unveiled in late December is an attack on ",
            "their sovereign authority regulating bodies of water and ",
            "surrounding land. The lawsuit named the Environmental Protection ",
            "Agency (EPA) and U.S. Army Corps of Engineers, the two agencies ",
            "that signed off on the rule, as defendants in the case.",
        ]
    )

    res = search_text(txt, load_nlp_pipeline)
    print(res)
    assert len(res) == 3
    assert False
