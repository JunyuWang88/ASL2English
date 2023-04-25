import spacy

nlp = spacy.load("./output/model-best")
doc = nlp("WATER DRINK I NEED")
print([(w.text, w.dep_,w.head) for w in doc])


doc = nlp("I LOVE ANYTHING HAVE CHOCOLATE INSIDE")
print([(w.text, w.dep_, w.head) for w in doc])
