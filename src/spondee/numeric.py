from spondee.schemas import Numeric


def extract_numbers(text: str, nlp_model):
    """Segment text into sentences and extract numeric values."""
    doc = nlp_model(text)

    found = []
    sentences = [s for s in doc.sentences]
    for i, sentence in enumerate(sentences):
        for token in sentence.tokens:
            for word in token._words:
                if word._xpos == "CD":
                    num = Numeric(
                        sidx=i,
                        text=word._text,
                        start_char=word._start_char,
                        end_char=word._end_char,
                    )
                    found.append(num)

    return found
