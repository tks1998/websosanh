from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
import contractions
import unicodedata
import inflect
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def remove_punctuation(text, remove_digit=False):
    pattern = r'[^a-zA-z0-9\s]'

    text = re.sub(pattern, '', text)
    return text


def remove_non_ascii(words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def remove_stopwords(words):
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def stem_words(words):
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def preprocessing(terms, remove_non_ACSII=True, lemmatize=True, stemming=True):
    if remove_non_ACSII:
        terms = remove_non_ascii(terms)
    terms = remove_stopwords(terms)
    if lemmatize:
        terms = lemmatize_verbs(terms)
    if stemming:
        terms = stem_words(terms)

    return terms


def get_terms(text, remove_punc=True, remove_non_ACSII=True, lemmatize=True, stemming=True):
    text = text.lower()
    if remove_punc:
        text = remove_punctuation(text)
    terms = nltk.word_tokenize(text)
    terms = preprocessing(terms, remove_non_ACSII=remove_non_ACSII,
                          lemmatize=lemmatize, stemming=stemming)
    return terms


def compute_idf(corpus):
    num_docs = len(corpus)
    idf = defaultdict(lambda: 0)
    for doc in corpus:
        for word in doc.keys():
            idf[word] += 1
    for word, value in idf.items():
        idf[word] = 1 + math.log(num_docs / value)
    return idf


def compute_tf(corpus):
    tf = copy.deepcopy(corpus)
    for doc in tf:
        for word, value in doc.items():
            doc[word] = value / len(doc)
    return tf


def compute_weight(tf, idf):
    weight = list()
    for doc in tf:
        weight_ = list()
        for term in idf.keys():
            weight_.append(doc[term] * idf[term] if term in doc.keys() else 0)
        weight.append(weight_)
        #print(len(weight))

    return weight
#=========================================================
corpus = list()
for doc in docs:
    terms = get_terms(headline)
    bag_of_words = Counter(terms)
    corpus.append(bag_of_words)
    #print(len(corpus))


idf = compute_idf(corpus)
tf = compute_tf(corpus)
represent_tfidf = weight.compute_weight(tf, idf)