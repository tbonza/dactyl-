"""
References:
    https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    https://catalog.ldc.upenn.edu/docs/LDC2011T03/treebank/english-treebank-guidelines-addendum.pdf
"""

from collections import deque
from typing import List, Tuple

import stanza

from spondee.schemas import Sentence


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


def identify_nounphrase(txt: List[Tuple[str, str]]) -> str:
    """Breadth-First Search concatenates text using grammatical rules."""
    np = []

    noun_tags = set(["NN", "NNS", "NNP", "NNPS"])
    if len(noun_tags & set([tag for tag, _ in txt])) == 0:
        return "", False

    q = deque(txt)
    while q:
        tag, s = q.popleft()
        if tag == "DT" or tag[:3] == "PRP":
            continue
            
        np.append((tag,s))
    return np, True


def nounphrase_text(node) -> Tuple[str, bool]:
    """Depth first search recovers child node and parent label."""
    txt = []
    stack = [node]
    prev_label = node.label
    while stack:
        node = stack.pop()
        if len(node.children) == 0:
            txt.append((prev_label, node.label))

        stack.extend(node.children)
        prev_label = node.label

    txt.reverse()  # word-order is left-to-right
    return identify_nounphrase(txt)


def extract_noun_phrases(verb_phrase):
    """Extract noun phrases from within a verb phrase using BFS."""
    noun_phrases = []
    q = deque([verb_phrase])
    while q:
        node = q.popleft()
        if node.label == "NP":
            txt, status = nounphrase_text(node)
            if status:
                noun_phrases.append(txt)

        else:
            q.extend(node.children)

    return noun_phrases


def identify_triplets(paths):
    compound = []
    for noun_phrase, verb_phrase in paths:
        np = extract_noun_phrases(noun_phrase)
        vp = extract_noun_phrases(verb_phrase)

        compound.append((np, vp))

    return compound


def extract_text(node):
    txt = []
    q = deque([node])
    while q:
        node = q.pop()
        if len(node.children) == 0:
            txt.append(node.label)

        q.extend(node.children)

    txt.reverse()
    return txt


def sentence_slots(compound_phrases, sidx: int):
    slots = []
    for noun_phrase, verb_phrase in compound_phrases:
        np = extract_noun_phrases(noun_phrase)
        vp = extract_noun_phrases(verb_phrase)

        sentence = Sentence(
            sidx=sidx,
            subject=np,
            predicate=vp,
            subject_text=extract_text(noun_phrase),
            predicate_text=extract_text(verb_phrase),
        )
        slots.append(sentence)

    return slots


def search_text(text: str, nlp_model):
    results = []

    doc = nlp_model(text)
    trees = [s.constituency for s in doc.sentences]
    for i, tree in enumerate(trees):
        paths = identify_statements(tree)
        slots = sentence_slots(compound_phrases=paths, sidx=i)
        results.extend(slots)

    return results
