import spacy
from spacy.tokens import Doc, DocBin

nlp = spacy.blank("en")
docbin = DocBin()
words = ['BUILDING', 'HISTORY', 'LEARN', 'ABOUT', 'I', 'LIKE']
heads = [0, 3, 3, 0, 5, 3]
deps = ['ROOT', 'nsubj', 'neg', 'conj', 'conj', 'dobj']
doc = Doc(nlp.vocab, words=words, heads=heads, deps=deps)
docbin.add(doc)
docbin.to_disk("./train.spacy")