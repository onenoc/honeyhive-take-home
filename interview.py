import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pyLDAvis
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Corpus class
class Corpus:
    def __init__(self, corpus, fields, title=None,n=3, k=3):
        '''
        @param corpus: a list of dictionaries
        @param fields: a list of strings that we want to extract from each dictionary in corpus
        @param n: the number of words we want to return for each topic
        @param k: the number of topics we want
        '''
        self.n = n
        self.k = k
        self.y = [d['accepted'] for d in corpus]
        self.y = np.array(self.y)
        self.title = title
        corpus = [{k: d[k] for k in fields} for d in corpus]
        self.corpus = [' '.join([d[k] for k in fields]) for d in corpus]
        self.corpus = self.preprocess_corpus()
        #get a matrix of word counts
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
        self.X = self.vectorizer.fit_transform(self.corpus)
        #create tf-idf matrix
        self.tfidf = TfidfTransformer()
        self.X_tfidf = self.tfidf.fit_transform(self.X).toarray()
        #get the vocabulary
        self.vocabulary = self.vectorizer.get_feature_names_out()
        #get indices
        indices = np.arange(len(self.corpus))
        #split train/test
        self.x_train, self.x_test, self.y_train, self.y_test, self.train_indices, self.test_indices = train_test_split(self.X_tfidf, self.y, indices, test_size=0.2)
    
    def preprocess_corpus(self):
        '''
        @summary: preprocess the corpus by removing punctuation, numbers, stopwords, and lemmatizing/stemming words
        '''
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
        '''
        @summary: use latent dirichlet allocation to find k topics and get the top n words for each topic
        '''
        #use latent dirichlet allocation to find topics
        self.lda = LatentDirichletAllocation(n_components=self.k, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(self.X)
        #get the top n words for each topic
        important_words = {}
        values = {}
        for topic_idx, topic in enumerate(self.lda.components_):
            #get the indices of the top n words for each topic
            word_idx = np.argsort(topic)[::-1][:self.n]
            #get the words at those indices
            important_words[topic_idx] = [self.vocabulary[i] for i in word_idx]
            values[topic_idx] = [topic[i] for i in word_idx]
        self.important_words = important_words
        self.values = values
        return important_words, values
    
    def plot_n_important_word_for_k_topics(self):
        '''
        @summary: plot the top n words for each topic in a bar chart
        '''
        keys = self.important_words.keys()
        for key in keys:
            plt.figure()
            plt.bar(self.important_words[key],self.values[key])
            plt.title(self.title+'Topic ' + str(key))
            plt.savefig(self.title+'Topic ' + str(key)+'.png')

    def logistic_regression(self):
        '''
        @summary: fit a logistic regression model to the data and return the accuracy on the test set
        '''
        #fit logistic regression model
        self.logistic = LogisticRegression(random_state=0).fit(self.x_train, self.y_train)
        #get accuracy
        return self.logistic.score(self.x_test, self.y_test)
    
    def words_influencing_result(self,idx,l=2):
        '''
        @summary: return the l words that most influence the result towards failure (even if it doesn't fail) of the idx'th entry in the test set
        '''
        #get the coefficients of the logistic regression model
        coefs = self.logistic.coef_[0]
        entry = self.x_test[idx]
        linear_predictor_terms = coefs * entry
        #get the l indices with the smallest linear predictor terms
        top_linear_predictor_indices = np.argsort(linear_predictor_terms)[:l]
        #get the words at those indices
        return [self.vocabulary[i] for i in top_linear_predictor_indices]

    def predict(self,idx):
        '''
        @summary: predict whether the idx'th entry in the test set is accepted or not
        '''
        #get the idx'th entry in the test set
        entry = self.x_test[idx]
        #predict whether the idx'th entry in the test set is accepted or not
        return self.logistic.predict(entry.reshape(1,-1))

    def predict_proba(self,idx):
        '''
        @summary: predict the probability that the idx'th entry in the test set is accepted
        '''
        #get the idx'th entry in the test set
        entry = self.x_test[idx]
        #predict the probability that the idx'th entry in the test set is accepted
        return self.logistic.predict_proba(entry.reshape(1,-1))

#read final_mle_dataset.json from file
with open('final_mle_dataset.json') as json_file:
    data = json.load(json_file)

#get data[0]'s relevant keys
inputs = ['product_name', 'product_description', 'prospect_name', 'prospect_industry', 'prospect_title']

#get entries of data where accepted is False
rejected = [d for d in data if d['accepted'] == False]

#1. Provide a topic analysis on what kinds of inputs and outputs the prompt template fails on

#create a corpus object from the rejected entries, inputs
corpus = Corpus(rejected, inputs,'Rejected: Inputs ',5,3)
corpus.get_top_n_words_for_k_topics()
corpus.plot_n_important_word_for_k_topics()

#create a corpus object from the rejected entries, email (outputs)
corpus = Corpus(rejected, ['email'],'Rejected: E-mail ',5,3)
corpus.get_top_n_words_for_k_topics()
corpus.plot_n_important_word_for_k_topics()

#create a corpus from all entries, email
corpus = Corpus(data, ['email'],5,3)
#fit logistic regression model, regressing email accepted/not on the tf-idf matrix
print(corpus.logistic_regression())

#Get the coefficients of the logistic regression model
coefs = corpus.logistic.coef_[0]
#Get the indices of the features with the largest coefficients
top_feature_indices = np.argsort(coefs)[:5]
important_words = [corpus.vocabulary[i] for i in top_feature_indices]
print('words with small logistic regression coefficients')
print(important_words)

#2.	Analyze the model outputs for problematic behaviors
#We output the words with the smallest (ideally negative with largest magnitude) linear predictor terms for the first 10 entries in the test set
#This is not a great measure of problematic behavior, as it seems to have nothing to do with the critiques.
for i in range(10):
    print('words with largest magnitude linear predictor terms for entry',i)
    print(corpus.words_influencing_result(i))
    print('actual result:',corpus.y_test[i])
    print('predicted result:',corpus.predict(i))
    print('')

#3. Suggest improvements to the original prompt template
#We suggest removing words that have the smallest logistic regression coefficients.
#Outside of the pure machine learning, we notice that most of the critiques have to do with lack of personalization
#The edited versions tend to include information about the propsect industry and prospect title. Thus we recommend including those.
for i in range(100):
    idx = corpus.test_indices[i]
    if corpus.y_test[i]==0:
        print('original e-mail')
        print(data[idx]['email'])
        print('we suggest removing the following words')
        print(corpus.words_influencing_result(i))
        print(data[idx]['critique'])
        print('we suggest including the following words to make it seem more personalized, if they arent yet included')
        print(data[idx]['prospect_industry'])
        print(data[idx]['prospect_title'])
        print('\n\n')


#4. Suggest evaluation criteria to compare prompt templates
#One simple approach is, given two prompt templates, use the probability output of logistic regression to determine which is more likely to be accepted
#The one with the higher probability is better in the sense that it is more likely to be accepted
#Here we predict, for the first ten entries, the probability of being accepted.
for i in range(10):
    print('Predicted probability of being accepted for entry',i)
    print(corpus.predict_proba(i)[0][1])
    print('actual result:',corpus.y_test[i])
    print('')


