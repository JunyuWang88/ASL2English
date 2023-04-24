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

lines = ['I like to ice skate on our pond',
         'OUR POND ICE SKATE I LIKE',
    "Don't forget to put your name on the paper.",
    "PAPER YOUR NAME WRITE NOT FORGET."]


with open("EngToASLPairs.txt", "r") as f:
    lines = f.readlines()


def preprocess_token(token_text):
    expanded_text = contractions.fix(token_text)
    token_text = re.sub(r'[^\w\s]', '', expanded_text)

    return token_text


''' Construct word to word mapping '''


def construct_token_level_mapping(asl_text, english_text):
    asl_to_eng = {}
    eng_to_asl = {}
    asl_tokens = asl_text.split()

    # print("asl whole text = {}".format(asl_text))
    for asl_token in asl_tokens:

        best_match = find_best_matching_english_token(asl_token, english_text.split())
        if best_match is not None:
            # print("asl = {}, english = {}".format(asl_token, best_match))
            asl_to_eng[asl_token] = best_match
            eng_to_asl[best_match] = asl_token

    return asl_to_eng, eng_to_asl


def find_root(current_english_root, eng_tokens, english_heads_words, eng_to_asl):
    index_ = eng_tokens.index(current_english_root)
    if current_english_root == english_heads_words[index_]:
        return current_english_root
    if current_english_root in eng_to_asl:
        return current_english_root
    else:
        return find_root(english_heads_words[index_], eng_tokens, english_heads_words, eng_to_asl)


def process_asl_heads(asl_text, english_text, english_heads, english_heads_words):
    # set up
    asl_to_eng, eng_to_asl = construct_token_level_mapping(asl_text, english_text)
    asl_tokens = asl_text.split()
    eng_tokens = english_text.split()
    #
    # print(
    #     "asl_to_eng = {}, eng_to_asl = {}, english_heads = {}, english_heads_words = {}".format(asl_to_eng, eng_to_asl,
    #                                                                                             english_heads,
    #                                                                                             english_heads_words))
    # print("asl_text = {}, english_text = {}".format(asl_text, english_text))
    asl_heads = []
    asl_heads_words = []
    for asl_token in asl_tokens:
        # if asl_token == "POND":
        #     print("here")
        if asl_token in asl_to_eng:
            index_ = eng_tokens.index(asl_to_eng[asl_token])
            current_english_root = english_heads_words[index_]
            root_english_head = find_root(current_english_root, eng_tokens, english_heads_words, eng_to_asl)

            if root_english_head in eng_to_asl:
                index_ = asl_tokens.index(eng_to_asl[root_english_head])
                asl_heads.append(index_)
                asl_heads_words.append(asl_tokens[index_])
            else:
                asl_heads.append('-')
                asl_heads_words.append('-')
        else:
            asl_heads.append('-')
            asl_heads_words.append('-')

    return asl_heads, asl_heads_words


def process_asl_deps(asl_text, english_text, english_deps):
    # set up
    asl_to_eng, eng_to_asl = construct_token_level_mapping(asl_text, english_text)
    asl_tokens = asl_text.split()
    eng_tokens = english_text.split()
    asl_deps = []
    for asl_token in asl_tokens:
        if asl_token in asl_to_eng:
            index_ = eng_tokens.index(asl_to_eng[asl_token])
            asl_deps.append(english_deps[index_])
        else:
            asl_deps.append('-')

    return asl_deps


'''Given an asl token, find the closest english token '''


def find_best_matching_english_token(asl_token_text, english_tokens_texts, threshold=1):
    min_distance = float('inf')
    best_match = None
    nlp_en = spacy.load("en_core_web_sm")
    # Lowercase and lemmatize ASL token
    asl_lemma = nlp_en(asl_token_text.lower())[0].lemma_

    for english_token_text in english_tokens_texts:
        # Lowercase and lemmatize English token
        english_lemma = nlp_en(english_token_text.lower())[0].lemma_

        distance = min(Levenshtein.distance(asl_lemma, english_lemma),
                       Levenshtein.distance(asl_token_text.lower(), english_lemma))
        if distance < min_distance:
            min_distance = distance
            best_match = english_token_text
    if min_distance <= threshold:
        return best_match
    else:
        return None


def preprocess_asl_english_data_pairs(lines):
    data = []
    num_processed_pairs = 0
    num_pairs = len(lines) / 2
    # Remove newline characters and filter out empty lines
    lines = [line.strip() for line in lines if line.strip()]
    # Load the English language model
    nlp = spacy.load("en_core_web_sm")
    # Group English and ASL lines into pairs
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            english_text = preprocess_token(lines[i])
            asl_text = preprocess_token(lines[i + 1])

            # =========== Process English ==================
            # Process English text with spaCy
            doc = nlp(english_text)
            # Print the dependency relations
            # for token in doc:
            #     print(f"{token.text:{10}} {token.dep_:{10}} {token.head.text}")
            # Extract head indices and dependency labels
            heads_words = [token.head.text for token in doc]
            heads = [token.head.i for token in doc]
            deps = [token.dep_ for token in doc]

            # =========== Process ASL =======================
            asl_heads, asl_heads_words = process_asl_heads(asl_text, english_text, heads, heads_words)
            asl_deps = process_asl_deps(asl_text, english_text, deps)
            if '-' in asl_heads:
                continue
            data_item = {
                "english": {
                    "text": english_text,
                    "heads_word": heads_words,
                    "heads": heads,
                    "deps": deps
                },
                "asl": {
                    "text": asl_text,
                    "heads_word": asl_heads_words,
                    "heads": asl_heads,
                    "deps": asl_deps
                }
            }

            data.append(data_item)
    return data


def transform_training_data(data):
    transformed_data = []

    for item in data:
        asl_text = item['asl']['text']
        asl_heads = item['asl']['heads']
        asl_deps = item['asl']['deps']

        transformed_data.append((asl_text, {'heads': asl_heads, 'deps': asl_deps}))

    save_to_json(transformed_data, "training_asl_data.json")

    return transformed_data


def save_to_json(data, filename):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def load_from_json(filename):
    with open(filename, 'r') as infile:
        data = json.load(infile)

    return data


raw_training_data = preprocess_asl_english_data_pairs(lines)
save_to_json(raw_training_data, "english_asl_pairs_raw_data.json")
print(raw_training_data)
transform_training_data(raw_training_data)
