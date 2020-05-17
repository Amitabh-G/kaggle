# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:35:55 2020

@author: amitabh.gunjan
"""
import logging
import pandas as pd
import numpy as np
from sklearn import feature_extraction, model_selection, decomposition

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC

#Create and configure logger 
def logger__():
    logging.basicConfig(filename="D:/other/study/ml-ai-dl-practice/nlp-stanford/kaggle/nlp-getting-started/logs/logs__.log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w') 
    #Creating an object 
    logger=logging.getLogger() 
    #Setting the threshold of logger to DEBUG 
    logger.setLevel(logging.DEBUG)
    return logger
    
#Load dataset
train = pd.read_csv("D:/other/study/ml-ai-dl-practice/nlp-stanford/kaggle/nlp-getting-started/data/train.csv")
test = pd.read_csv("D:/other/study/ml-ai-dl-practice/nlp-stanford/kaggle/nlp-getting-started/data/test.csv")

def generate_cooccurence_matrix(train, fraction):
    train = train.sample(frac=fraction, replace=True, random_state=17)
    print(train.shape)
    corpus = train['text']
    # vectorizer = feature_extraction.text.CountVectorizer()
    vectorizer = feature_extraction.text.TfidfVectorizer()
    counts = vectorizer.fit_transform(corpus)
    counts.toarray()
    counts_dense = counts.toarray()
    print(counts_dense.shape)
    co_occurence = (counts.T * counts) # this is co-occurrence matrix.
    co_occurence.setdiag(0) # fill same word cooccurence to 0
    # print("The co-occurence values in matrix format:\n",co_occurence.todense())
    print("The co-occurence matrix shape:\n", co_occurence.shape)
    centered_co_occurence = co_occurence - np.mean(co_occurence, axis=0)
    vocab = vectorizer.vocabulary_    
    return centered_co_occurence, vocab, vectorizer, train

def generate_latent_variables(centered_co_occurence, num_components):
    
    normalizer = Normalizer()
    normalizer.fit_transform(centered_co_occurence)
    pca = decomposition.PCA(svd_solver='randomized', random_state=17)
    pca.fit(centered_co_occurence)
    components = pca.components_
    k_components = components[:num_components]
    latent_vars = k_components*centered_co_occurence
    latent_vars_matrix = latent_vars.T
    return k_components, latent_vars_matrix, normalizer
    
def generate_sent_2_vec(train_data, vocab, latent_vars_matrix, num_components, vectorizer):
    tokenizer = vectorizer.build_tokenizer()
    sent_vector_matrix = np.zeros(shape=(train_data.shape[0], num_components))
    # print("train data shape---",train_data.shape[0])
    train_data_idx = train_data.index
    for i in range(len(train_data_idx)):
        sent_vector = np.zeros(shape=(num_components,))
        doc = train_data['text'][train_data_idx[i]]
        # print("The doc being tokenized is:", doc)
        doc = str(doc)
        tokens = tokenizer(doc)
        ctr = 0
        for token in tokens:
            if token in vocab:
                ctr += 1
                idx = vocab[token]
                word_vector = latent_vars_matrix[idx:idx+1]
                sent_vector = np.add(sent_vector, word_vector)
        if ctr > 0:
            sent_vector = sent_vector/ctr
        else:
            pass
        sent_vector_matrix[i] = sent_vector
    return sent_vector_matrix

def nan_to_zero(sent_vector_matrix):
    where_are_NaNs = np.isnan(sent_vector_matrix)
    sent_vector_matrix[where_are_NaNs] = 0
    return sent_vector_matrix
    
def genrate_test_matrix(test, vectorizer):
    test_corpus = test['text']
    counts_test = vectorizer.transform(test_corpus)
    co_occurence = (counts_test.T * counts_test) # this is co-occurrence matrix.
    co_occurence.setdiag(0) # fill same word cooccurence to 0
    # print("The co-occurence values in matrix format:\n",co_occurence.todense())
    print("The co-occurence matrix shape:\n", co_occurence.shape)
    centered_co_occurence = co_occurence - np.mean(co_occurence, axis=0)
    return centered_co_occurence

def get_latent_vars_test(test_matrix, k_components, normalizer):
    normalizer.transform(test_matrix)
    latent_vars = k_components*centered_co_occurence
    latent_vars_matrix = latent_vars.T
    return latent_vars_matrix

#########
### Model
#########

classifiers = (RidgeClassifier(), )
fraction = 0.01
centered_co_occurence, vocab, vectorizer, train_data = generate_cooccurence_matrix(train, fraction)
num_components = (70,) 
for clf in classifiers:
    for i in num_components:
        logger = logger__()
        logging.info(f"The classifer trained is:, {clf}")
        print("Dimension of word vectors: ", i)
        logging.info(f"The number of eigenvectors used are, {i}")
        k_components, latent_vars_matrix, normalizer = generate_latent_variables(centered_co_occurence, i)
        sent_vector_matrix = generate_sent_2_vec(train_data, vocab, latent_vars_matrix, i, vectorizer)
        sent_vector_matrix = nan_to_zero(sent_vector_matrix)
        logging.info("Started the cross validation procedure.")
        scores = model_selection.cross_val_score(clf, sent_vector_matrix, train_data["target"], cv=5, scoring="f1")
        logging.info(f"The scores on cv set:, {scores}")
        print("The scores on cv set: ",scores)
        print("Average score: ", np.mean(scores))
        logging.info(f"Average score:, {np.mean(scores)}")
        Y = train_data['target']
        clf.fit(sent_vector_matrix, Y)
        
        test_cooccur_matrix = genrate_test_matrix(test, vectorizer)
        latent_vars_test = get_latent_vars_test(test_cooccur_matrix, k_components, normalizer)
        sent_matrix_test = generate_sent_2_vec(test, vocab, latent_vars_test, i, vectorizer)
        sample_submission = pd.read_csv("D:/other/study/ml-ai-dl-practice/nlp-stanford/kaggle/nlp-getting-started/data/sample_submission.csv")
        sample_submission["target"] = clf.predict(sent_matrix_test)
        submission_file_ext_tfidf = "submission" + "_" + str(fraction) + "_" + str(i) + "_tfidf"
        submission_file = "D:/other/study/ml-ai-dl-practice/nlp-stanford/kaggle/nlp-getting-started/data/" + submission_file_ext_tfidf + ".csv"
        sample_submission.to_csv(submission_file, index=False)
        