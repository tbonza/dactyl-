import pytest

from spondee.search import nlp_pipeline


@pytest.fixture
def load_nlp_pipeline():
    return nlp_pipeline()
