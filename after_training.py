import spacy

nlp = spacy.load("./output/model-best")
doc = nlp("Leetcode I need")
print([(w.text, w.dep_,w.head, w.pos_) for w in doc])


# nlp = spacy.load("en_core_web_sm")



# Dependency parse as a list of tuples
asl_dependency = [('You', 'nsubj', 'MEET'), ('MEET', 'ROOT', 'MEET'), ('MY', 'poss', 'BROTHER'), ('BROTHER', 'dobj', 'MEET'), ('YOU', 'nsubj', 'MEET')]

# Create an empty Doc object with specified words and spaces
words = [token for token, _, _ in asl_dependency]
spaces = [True] * len(words)
doc = spacy.tokens.Doc(nlp.vocab, words=words, spaces=spaces)

# Create a mapping of token text to their corresponding position in the doc
token_mapping = {token.text: i for i, token in enumerate(doc)}

# Set the dependency and head information
for token, dep, head in asl_dependency:
    t = doc[token_mapping[token]]
    t.dep_ = dep
    t.head = doc[token_mapping[head]]

# Generate the sentence from the dependency parse
sentence = " ".join([token.text for token in doc if not token.dep_ == "ROOT"]).strip()
print(sentence)