from spondee.numeric import extract_numbers


import pytest


@pytest.fixture
def example_numeric_text():
    return "We saw a 20% increase in 2024. As we wind on down the road."


def test_part_of_speech_tags(load_nlp_pipeline, example_numeric_text):
    output = extract_numbers(example_numeric_text, load_nlp_pipeline)

    s = example_numeric_text
    assert len(output) == 2
    assert all([d.sidx == 0 for d in output])
    assert s[output[0].start_char : output[0].end_char] == output[0].text
    assert s[output[1].start_char : output[1].end_char] == output[1].text
