from spondee.search import (
    identify_statements,
    extract_noun_phrases,
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

    assert res[1].subject == ["Gavin Sheets"]
    assert res[1].predicate == ["first career grand slam"]

    assert res[0].subject == ["Chicago White Sox"]
    assert res[0].predicate[-2] == "Boston Red Sox"


def test_part_of_speech_tags(load_nlp_pipeline, example_text):
    nlp = load_nlp_pipeline
    doc = nlp(example_text)

    sidx = 0
    tree = doc.sentences[sidx].constituency

    paths = identify_statements(tree)
    np, vp = paths[0]
    output_vp = extract_noun_phrases(vp)
    output_np = extract_noun_phrases(np)

    assert output_vp[-2] == "Boston Red Sox"
    assert output_np[0] == "Chicago White Sox"


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
