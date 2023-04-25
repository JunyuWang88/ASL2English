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
import json
import glob


def read_and_append_json_files(pattern):
    all_data = []

    for file_name in glob.glob(pattern):
        with open(file_name, 'r') as f:
            data = json.load(f)
            all_data.extend(data)

    return all_data


def transform_training_data(data):
    transformed_data = []

    for item in data:
        asl_text = item['asl']['text']
        asl_heads = item['asl']['heads']
        asl_deps = item['asl']['deps']

        if '-' in asl_heads:
            continue

        transformed_data.append((asl_text, {'heads': asl_heads, 'deps': asl_deps}))

    return transformed_data


# Load the data from the JSON file
def load_from_json(filename):
    with open(filename, 'r') as infile:
        data = json.load(infile)

    return data


# Read and append all JSON files
pattern = "english_asl_pairs_raw_*_data.json"
all_data = read_and_append_json_files(pattern)

# Transform the data
TRAINING_DATA = transform_training_data(all_data)
TRAINING_DATA = TRAINING_DATA[5:10]

PARSER_CONFIG = 'parser.cfg'


def create_training_examples(training_data: List[Tuple]) -> List[Example]:
    """ Create list of training examples """
    examples = []
    nlp = spacy.load('en_core_web_sm')
    for text, annotations in training_data:
        # print(f"{text} - {annotations}")
        try:
            examples.append(Example.from_dict(nlp(text), annotations))
        except Exception as e:
            print(f"Error processing: {text} - {annotations}")
            continue
    print("finish")
    return examples


def save_trained_nlp(nlp, custom_name):
    nlp.to_disk(custom_name)


def load_trained_nlp(custom_name):
    nlp = spacy.load('en_core_web_sm', exclude=["parser"])
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
