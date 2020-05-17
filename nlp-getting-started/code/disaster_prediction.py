# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:18:14 2020

@author: amitabh.gunjan
"""
import pandas as pd
import numpy as np
from sklearn import feature_extraction, linear_model, model_selection, preprocessing, decomposition

train = pd.read_csv("D:/other/study/ml-ai-dl-practice/nlp-stanford/kaggle/nlp-getting-started/data/train.csv")
print(train.shape)
test = pd.read_csv("D:/other/study/ml-ai-dl-practice/nlp-stanford/kaggle/nlp-getting-started/data/test.csv")
# print(train[train["target"] == 0]["text"])
# print(train['text'])
corpus = train['text']
test_corpus = test['text']

vectorizer = feature_extraction.text.CountVectorizer()
counts = vectorizer.fit_transform(corpus)
counts_test = vectorizer.transform(test_corpus)
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, counts, train["target"], cv=3, scoring="f1")
# print(scores)


clf.fit(counts, train["target"])
sample_submission = pd.read_csv("D:/other/study/ml-ai-dl-practice/nlp-stanford/kaggle/nlp-getting-started/data/sample_submission.csv")


sample_submission["target"] = clf.predict(counts_test)
sample_submission.head()
sample_submission.to_csv("D:/other/study/ml-ai-dl-practice/nlp-stanford/kaggle/nlp-getting-started/data/submission.csv", index=False)
