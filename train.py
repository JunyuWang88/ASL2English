import spacy
import random
import json
import string
import re
import contractions
from spacy.tokens import Doc
from spacy.training import Example
from spacy.pipeline import DependencyParser
from typing import List, Tuple
import Levenshtein


# Load the data from the JSON file
def load_from_json(filename):
    with open(filename, 'r') as infile:
        data = json.load(infile)

    return data


TRAINING_DATA = load_from_json('training_asl_data.json')
print(TRAINING_DATA)

PARSER_CONFIG = 'parser.cfg'


def create_training_examples(training_data: List[Tuple]) -> List[Example]:
    """ Create list of training examples """
    examples = []
    nlp = spacy.load('en_core_web_sm')
    for text, annotations in training_data:
        print(f"{text} - {annotations}")
        examples.append(Example.from_dict(nlp(text), annotations))
    return examples


def save_trained_nlp(nlp, custom_name):
    nlp.to_disk(custom_name)


def load_trained_nlp(custom_name):
    nlp = spacy.load('en_core_web_md', exclude=["parser"])
    parser_nlp = spacy.load(custom_name)
    nlp.add_pipe("parser", source=parser_nlp)
    return nlp


nlp = spacy.blank('en')
# Create new parser
parser = nlp.add_pipe('parser', first=True)
for text, annotations in TRAINING_DATA:
    for label in annotations['deps']:
        if label not in parser.labels:
            parser.add_label(label)
print(f"Added labels: {parser.labels}")

examples = create_training_examples(TRAINING_DATA)

# Training
# NOTE: The 'lambda: examples' part is mandatory in Spacy 3 - https://spacy.io/usage/v3#migrating-training-python
optimizer = nlp.initialize(lambda: examples)
print(f"Training ... ", end='')
for i in range(25):
    print(f"{i} ", end='')
    random.shuffle(examples)
    nlp.update(examples, sgd=optimizer)
print(f"... DONE")

save_trained_nlp(nlp, "new_parser")

nlp = load_trained_nlp("new_parser")

doc = nlp(u'find a high paid job with no degree')
print(f"Arcs: {[(w.text, w.dep_, w.head.text) for w in doc if w.dep_ != '-']}")
