"""
References:
    https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    https://catalog.ldc.upenn.edu/docs/LDC2011T03/treebank/english-treebank-guidelines-addendum.pdf
"""

from collections import deque
from typing import List

import stanza

from spondee.schemas import LeafLabel, Sentence, SentenceMetadata


def nlp_pipeline():
    nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,constituency")
    return nlp


def has_npvp(children):
    labels = [c.label for c in children]
    if "NP" in labels and "VP" in labels:
        return True

    return False


def identify_statements(tree):
    stack = [tree]
    paths = []

    while stack:
        node = stack.pop()

        if node.label == "S" and has_npvp(node.children):
            kids = {child.label: child for child in node.children}
            noun_phrase = kids["NP"]
            verb_phrase = kids["VP"]
            paths.append((noun_phrase, verb_phrase))

        stack.extend(node.children)

    return paths


def extract_node_label(node):
    path = []
    stack = [node]
    prev_label = None
    while stack:
        node = stack.pop()
        if len(node.children) == 0:
            path.append((prev_label, node.label))

        stack.extend(node.children)
        prev_label = node.label

    path.reverse()
    return path


def extract_noun_phrases(node):
    paths = []
    q = deque([node])
    while q:
        node = q.popleft()
        if node.label == "NP":
            paths.append(extract_node_label(node))

        else:
            q.extend(node.children)

    return paths


def filter_noun_phrases(extracted_np):
    noun_tags = set(["NN", "NNS", "NNP", "NNPS"])
    extract_tags = set([t for t, _ in extracted_np])

    if len(noun_tags & extract_tags) == 0:
        return []

    first_tag, _ = extracted_np[0]
    if first_tag == "DT" or first_tag[:3] == "PRP":
        return extracted_np[1:]

    return extracted_np


def nounphrase_metadata(npm, extract):
    meta_q = deque(npm)
    extract_q = deque(extract)

    found = []
    while extract_q:
        tag, s = extract_q.popleft()

        while meta_q:
            _meta = meta_q.popleft()
            if _meta["xpos"] == tag and _meta["text"] == s:
                found.append(_meta)
                break

    return found


def sentence_metadata(sidx: int, statements, simple_sentence: List[dict]):
    paths = []
    stack = statements
    q = deque(simple_sentence)
    while stack:
        noun_phrase, verb_phrase = stack.pop()

        _tagged_np = [filter_noun_phrases(r) for r in extract_noun_phrases(noun_phrase)]
        _tagged_vp = [filter_noun_phrases(r) for r in extract_noun_phrases(verb_phrase)]

        npq = deque(noun_phrase.leaf_labels())
        vpq = deque(verb_phrase.leaf_labels())

        _np = []
        _vp = []

        while npq or vpq:
            _leaf = q.popleft()

            if npq and npq[0] == _leaf["text"]:
                _np.append(_leaf)
                npq.popleft()

            elif vpq and vpq[0] == _leaf["text"]:
                _vp.append(_leaf)
                vpq.popleft()

        _found_np = []
        for tagged in _tagged_np:
            _meta = nounphrase_metadata(_np, tagged)
            if len(_meta) > 0:
                _found_np.append(_meta)

        _found_vp = []
        for tagged in _tagged_vp:
            _meta = nounphrase_metadata(_vp, tagged)
            if len(_meta) > 0:
                _found_vp.append(_meta)

        _sentence_meta = SentenceMetadata(
            sidx=sidx,
            subject=[LeafLabel.model_validate(m) for m in _np],
            predicate=[LeafLabel.model_validate(m) for m in _vp],
            subject_noun_phrases=[
                [LeafLabel.model_validate(m) for m in item] for item in _found_np
            ],
            predicate_noun_phrases=[
                [LeafLabel.model_validate(m) for m in item] for item in _found_vp
            ],
        )

        paths.append(_sentence_meta)

    return paths


# def sentence_slots(compound_phrases, sidx: int):
#    slots = []
#    for noun_phrase, verb_phrase in compound_phrases:
#        np = extract_noun_phrases(noun_phrase)
#        vp = extract_noun_phrases(verb_phrase)
#
#        sentence = Sentence(
#            sidx=sidx,
#            subject=np,
#            predicate=vp,
#            subject_text=extract_text(noun_phrase),
#            predicate_text=extract_text(verb_phrase),
#        )
#        slots.append(sentence)
#
#    return slots
#
#
# def search_text(text: str, nlp_model):
#    results = []
#
#    doc = nlp_model(text)
#    trees = [s.constituency for s in doc.sentences]
#    for i, tree in enumerate(trees):
#        paths = identify_statements(tree)
#        slots = sentence_slots(compound_phrases=paths, sidx=i)
#        results.extend(slots)
#
#    return results
