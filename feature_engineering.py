import os
import re
import nltk
import string
import numpy as np
import pandas as pd
from sklearn import feature_extraction
from tqdm import tqdm
import gensim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from functools import reduce

_wnl = nltk.WordNetLemmatizer()
token_pattern = r'(?u)\b\w\w+\b'
stopwords = set(nltk.corpus.stopwords.words('english'))
english_stemmer = nltk.stem.SnowballStemmer('english')

def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def gen_or_load_feats(feat_fn, headlines, bodies, feature_file):
    if not os.path.isfile(feature_file):
        feats = feat_fn(headlines, bodies)
        np.save(feature_file, feats)

    return np.load(feature_file)


def preprocess_data(line,
                    token_pattern=token_pattern,
                    exclude_stopword=True,
                    stem=True):
    token_pattern = re.compile(token_pattern, flags = re.UNICODE)
    tokens = [x.lower() for x in token_pattern.findall(line)]
    tokens_stemmed = tokens
    if stem:
        tokens_stemmed = stem_tokens(tokens, english_stemmer)
    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]
    return tokens_stemmed

def cosine_sim(x, y):
    try:
        if type(x) is np.ndarray: x = x.reshape(1, -1) # get rid of the warning
        if type(y) is np.ndarray: y = y.reshape(1, -1)
        d = cosine_similarity(x, y)
        d = d[0][0]
    except:
        print (x)
        print (y)
        d = 0.
    return d

def gen_word2vec_feats(headlines, bodies):
    #for idx, headline in enumerate(headlines):
        #headlines[idx] = clean(headline)
    #for idx, body in enumerate(bodies):
        #bodies[idx] = clean(body)

    d = { 'Headline': headlines, 'articleBody': bodies}
    df = pd.DataFrame(data=d)
    print ('generating word2vec features')
    df["Headline_unigram_vec"] = df["Headline"].map(lambda x: preprocess_data(x, exclude_stopword=False, stem=False))
    df["articleBody_unigram_vec"] = df["articleBody"].map(lambda x: preprocess_data(x, exclude_stopword=False, stem=False))

    #headline_unigram = []
    #for headline in df["Headline"]:
        #headline_unigram.append(ngrams(headline, 1))

    #body_unigram = []
    #for body in df["articleBody"]:
        #body_unigram.append(ngrams(body, 1))

    #df["Headline_unigram_vec"] = headline_unigram;
    #df["articleBody_unigram_vec"] = body_unigram;

    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    print ('model loaded')

    Headline_unigram_array = df['Headline_unigram_vec'].values
    headlineVec = map(lambda x: reduce(np.add, [model[y] for y in x if y in model], [0.]*300), Headline_unigram_array)
    headlineVec = list(headlineVec)
    #print ('headlineVec:')
    #print (headlineVec)
    #print ('type(headlineVec)')
    #print (type(headlineVec))
    #headlineVec = np.exp(headlineVec)
    headlineVec = normalize(headlineVec)
    #headlineVec = np.hstack(headlineVec)



    Body_unigram_array = df['articleBody_unigram_vec'].values
    bodyVec = map(lambda x: reduce(np.add, [model[y] for y in x if y in model], [0.]*300), Body_unigram_array)
    bodyVec = list(bodyVec)
    bodyVec = normalize(bodyVec)
    #bodyVec = np.hstack(bodyVec)

    # compute cosine similarity between headline/body word2vec features
    #simVec = []
    #for h in headlineVec:
        #for b in bodyVec:
            #simVec.append(cosine_sim(h,b))
    simVec = list(map(cosine_sim, headlineVec, bodyVec))
    return headlineVec, bodyVec, simVec

def word_overlap_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(features)
    return X


def refuting_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _refuting_words]
        X.append(features)
    return X


# Checks presence of a Question mark in the headline
def question_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        if '?' in headline:
            X.append(1)
        else:
            X.append(0)
    return X

# Count of capital letters in the body
def punctuation_features(headlines, bodies):
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        count = 0
        for c in body:
            if c in letters:
                count+=1
        X.append(count)
    return X


# Count of punctuation in the body
def cap_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        count = 0
        for c in body:
            if c in string.punctuation:
                count+=1
        X.append(count)
    return X


def supporting_features(headlines, bodies):
    _supporting_words = [
        'real',
        'yes',
        'true',
        'support',
        'good',
        'agree',
        'yeah',
        'approve',
        'advocate',
        'endorse',
        'uphold'
    ]
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _supporting_words]
        X.append(features)
    return X


def polarity_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        features.append(calculate_polarity(clean_headline))
        features.append(calculate_polarity(clean_body))
        X.append(features)
    return np.array(X)


def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features


def hand_features(headlines, bodies):

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        X.append(binary_co_occurence(headline, body)
                 + binary_co_occurence_stops(headline, body)
                 + count_grams(headline, body))


    return X
