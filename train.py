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
import spacy
from spacy.tokens import Doc, DocBin


def convert_train_and_validation_spacy(TRAINING_DATA, train_ratio=0.8):
    nlp = spacy.blank("en")
    # Shuffle the data
    random.shuffle(TRAINING_DATA)

    # Calculate the split index
    split_index = int(train_ratio * len(TRAINING_DATA))

    # Split the data into training and development sets
    train_data = TRAINING_DATA[:split_index]
    dev_data = TRAINING_DATA[split_index:]

    # Save the training data
    train_docbin = DocBin()
    for text, annotations in train_data:
        words = text.split()
        heads = annotations['heads']
        deps = annotations['deps']
        doc = Doc(nlp.vocab, words=words, heads=heads, deps=deps)
        train_docbin.add(doc)
    train_docbin.to_disk("./train.spacy")

    # Save the development data
    dev_docbin = DocBin()
    for text, annotations in dev_data:
        words = text.split()
        heads = annotations['heads']
        deps = annotations['deps']
        doc = Doc(nlp.vocab, words=words, heads=heads, deps=deps)
        dev_docbin.add(doc)
    dev_docbin.to_disk("./dev.spacy")


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

        num_of_root = 0
        for deps in asl_deps:
            if deps == 'ROOT':
                num_of_root += 1
        if num_of_root > 1:
            print("More than one root")
            print(asl_text)
            continue

        def has_cycle(graph):
            n = len(graph)
            visited = [False] * n
            rec_stack = [False] * n
            def dfs(v):
                visited[v] = True
                rec_stack[v] = True

                next_v = graph[v]
                if next_v != v and next_v != '-':  # exclude self-loop
                    if not visited[next_v]:
                        if dfs(next_v):
                            return True
                    elif rec_stack[next_v]:
                        return True

                rec_stack[v] = False
                return False

            for i in range(n):
                if not visited[i]:
                    if dfs(i):
                        return True

            return False
        if has_cycle(asl_heads):
            continue

        asl_heads = [x if x != '-' else None for x in asl_heads]

        transformed_data.append((asl_text, {'heads': asl_heads, 'deps': asl_deps}))

    print(len(transformed_data))

    return transformed_data


# Load the data from the JSON file
def load_from_json(filename):
    with open(filename, 'r') as infile:
        data = json.load(infile)

    return data


# Read and append all JSON files
pattern = "english_asl_pairs_*_data.json"
all_data = read_and_append_json_files(pattern)

# Transform the data
TRAINING_DATA = transform_training_data(all_data)


convert_train_and_validation_spacy(TRAINING_DATA)
#
# PARSER_CONFIG = 'parser.cfg'
#
#
# def create_training_examples(training_data: List[Tuple]) -> List[Example]:
#     """ Create list of training examples """
#     examples = []
#     nlp = spacy.load('en_core_web_sm')
#     for text, annotations in training_data:
#         # print(f"{text} - {annotations}")
#         try:
#             examples.append(Example.from_dict(nlp(text), annotations))
#         except Exception as e:
#             print(f"Error processing: {text} - {annotations}")
#             continue
#     print("finish")
#     return examples
#
#
# def save_trained_nlp(nlp, custom_name):
#     nlp.to_disk(custom_name)
#
#
# def load_trained_nlp(custom_name):
#     nlp = spacy.load('en_core_web_sm', exclude=["parser"])
#     parser_nlp = spacy.load(custom_name)
#     nlp.add_pipe("parser", source=parser_nlp)
#     return nlp
#
#
# nlp = spacy.blank('en')
# # Create new parser
# parser = nlp.add_pipe('parser', first=True)
# for text, annotations in TRAINING_DATA:
#     for label in annotations['deps']:
#         if label not in parser.labels:
#             parser.add_label(label)
# print(f"Added labels: {parser.labels}")
#
# examples = create_training_examples(TRAINING_DATA)
#
# # Training
# # NOTE: The 'lambda: examples' part is mandatory in Spacy 3 - https://spacy.io/usage/v3#migrating-training-python
# optimizer = nlp.initialize(lambda: examples)
# print(f"Training ... ", end='')
# for i in range(25):
#     print(f"{i} ", end='')
#     random.shuffle(examples)
#     nlp.update(examples, sgd=optimizer)
# print(f"... DONE")
#
# save_trained_nlp(nlp, "new_parser")
#
# nlp = load_trained_nlp("new_parser")
#
# doc = nlp(u'find a high paid job with no degree')
# print(f"Arcs: {[(w.text, w.dep_, w.head.text) for w in doc if w.dep_ != '-']}")
