import spacy
from spacy.lang.es.stop_words import STOP_WORDS
import es_core_news_sm
nlp = es_core_news_sm.load()
#nlp = spacy.load("es_core_news_sm", disable=['ner', 'parser', 'tagger'])

def clean(word):
    w = word
    while len(w) >= 1 and w[0] in ['/', '-', '¡', '¿', '.', ',', ';', ':', '\'', '"', '?', '!']:
        w = w[1:]
    while len(w) >= 1 and w[-1] in ['/', '-', '¡', '¿', '.', ',', ';', ':', '\'', '"', '?', '!']:
        w = w[:-1]
    return w

# Tokenizer based on spacy only
def tokenizer(doc, lowercase=True):
    return [x.orth_ for x in nlp(doc.lower() if lowercase else doc)]

# Tokenizing and deliting stopwords
def tokenizer_wo_stopwords(doc, lowecase=True):
    return [x.orth_ for x in nlp(doc.lower() if lowercase else doc) if x.orth_ not in STOP_WORDS]

# Tokenizing and lematizing
def tokenizer_with_lemmatization(doc, lowercase=True):
    return [clean(x.lemma_) for x in nlp(doc.lower() if lowercase else doc)]

# Tokenizing and lematizing delting stopwords
def tokenizer_with_lemmatization_wo_stopwords(doc, lowercase=True):
    return [clean(x.lemma_) for x in nlp(doc.lower() if lowercase else doc) if x.orth_ not in STOP_WORDS]

# Tokenizing and stemming
def tokenizer_with_stemming(doc, lowercase=True):
    stemmer = SnowballStemmer('spanish')
    return [stemmer.stem(word) for word in [x.orth_ for x in nlp(doc.lower() if lowercase else doc)]]