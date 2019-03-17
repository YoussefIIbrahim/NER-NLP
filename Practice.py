# import nltk
# from nltk.corpus import state_union
# from nltk.tokenize import PunktSentenceTokenizer
#
# train_text = state_union.raw("2005-GWBush.txt")
# sample_text = state_union.raw("2006-GWBush.txt")
#
# custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
#
# tokenized = custom_sent_tokenizer.tokenize("Lucas bought a house and he paid 3000 euros for that house.")
#
# def process_content():
#     try:
#         for i in tokenized:
#             print(i)
#             words = nltk.word_tokenize(i)
#             print(words)
#             tagged = nltk.pos_tag(words)
#             print(tagged)
#             namedEnt = nltk.ne_chunk(tagged)
#             for chunk in namedEnt:
#                 print(chunk.label())
#             namedEnt.draw()
#     except Exception as e:
#         print(str(e))
#
#
# process_content()

import nltk

doc = '''Andrew Yan-Tak Ng is a Chinese American computer scientist.
He is the former chief scientist at Baidu, where he led the company's
Artificial Intelligence Group. He is an adjunct professor (formerly 
associate professor) at Stanford University. Ng is also the co-founder
and chairman at Coursera, an online education platform. Andrew was born
in the UK in 1976. His parents were both from Hong Kong.'''
# tokenize doc
tokenized_doc = nltk.word_tokenize(doc)

# tag sentences and use nltk's Named Entity Chunker
tagged_sentences = nltk.pos_tag(tokenized_doc)
ne_chunked_sents = nltk.ne_chunk(tagged_sentences)

# extract all named entities
named_entities = []
for tagged_tree in ne_chunked_sents:
    if hasattr(tagged_tree, 'label'):
        entity_name = ' '.join(c[0] for c in tagged_tree.leaves())  #
        entity_type = tagged_tree.label()  # get NE category
        named_entities.append((entity_name, entity_type))
print(named_entities)
