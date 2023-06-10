import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#make a class that initializes with a corpus and has methods to preprocess the corpus and get the top n words for k topics
class Corpus:
    def __init__(self, corpus, fields, n=3, k=3):
        '''
        @param corpus: a list of dictionaries
        @param fields: a list of strings that we want to extract from each dictionary in corpus
        @param n: the number of words we want to return for each topic
        @param k: the number of topics we want
        '''
        self.n = n
        self.k = k
        corpus = [{k: d[k] for k in fields} for d in corpus]
        self.corpus = [' '.join([d[k] for k in fields]) for d in corpus]
    
    def preprocess_corpus(self):
        #remove punctuation
        corpus = [d.lower() for d in self.corpus]
        #remove numbers
        corpus = [re.sub(r'[^\w\s]','',d) for d in corpus]
        #remove stopwords
        corpus = [re.sub(r'\d+','',d) for d in corpus]
        stop_words = set(stopwords.words('english'))
        corpus = [' '.join([w for w in d.split() if not w in stop_words]) for d in corpus]
        #stem words
        stemmer = SnowballStemmer("english")
        corpus = [' '.join([stemmer.stem(w) for w in d.split()]) for d in corpus]
        #lemmatize words
        lemmatizer = WordNetLemmatizer()
        corpus = [' '.join([lemmatizer.lemmatize(w) for w in d.split()]) for d in corpus]
        return corpus
    
    def get_top_n_words_for_k_topics(self):
        #get a matrix of word counts
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(self.corpus)
        #get the vocabulary
        vocabulary = vectorizer.get_feature_names_out()
        #use latent dirichlet allocation to find topics
        lda = LatentDirichletAllocation(n_components=self.k, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(X)
        #get the top n words for each topic
        important_words = {}
        for topic_idx, topic in enumerate(lda.components_):
            #get the indices of the top n words for each topic
            word_idx = np.argsort(topic)[::-1][:self.n]
            #get the words at those indices
            important_words[topic_idx] = [vocabulary[i] for i in word_idx]
        return important_words

#read final_mle_dataset.json from file
with open('final_mle_dataset.json') as json_file:
    data = json.load(json_file)

#get data[0]'s relevant keys
inputs = ['product_name', 'product_description', 'prospect_name', 'prospect_industry', 'prospect_title']

#get entries of data where accepted is False
rejected = [d for d in data if d['accepted'] == False]

#create a corpus object from the rejected entries, inputs
corpus = Corpus(rejected, inputs,5,3)
rejected_inputs = corpus.preprocess_corpus()
important_words_inputs = corpus.get_top_n_words_for_k_topics()

#create a corpus object from the rejected entries, email
corpus = Corpus(rejected, ['email'],5,3)
rejected_email = corpus.preprocess_corpus()
important_words_email = corpus.get_top_n_words_for_k_topics()

#print the important words for each topic, both inputs and email
print(important_words_inputs)
print(important_words_email)
