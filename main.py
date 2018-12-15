#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import sys
import pandas as pd
import os
from scipy.spatial.distance import cdist
import json
import numpy as np
import indicoio
indicoio.config.api_key = "cbe0522905613dd0a2c4ff6442355e14"


# In[ ]:


# Question Answering

def make_feats(data):
    """
    Send our text data throught the indico API and return each text example's text vector representation
    """
    # TODO
    chunks = [data[x:x+100] for x in range(0, len(data), 100)]
    feats = []

    # working with chunks of the data at a time
    for chunk in chunks:
        feats.extend(indicoio.text_features(chunk))

    return feats

def calculate_distances(feats):
    # cosine distance is the most reasonable metric for comparison of these 300d vectors
    distances = cdist(feats, feats, 'cosine')
    return distances

def similarity_text(idx, distance_matrix, data, faqs, n_similar=5):
    """
    idx: the index of the text we're looking for similar questions to
         (data[idx] corresponds to the actual text we care about)
    distance_matrix: an m by n matrix that stores the distance between
                     document m and document n at distance_matrix[m][n]
    data: a flat list of text data
    """

    # these are the indexes of the texts that are most similar to the text at data[idx]
    # note that this list of 10 elements contains the index of text that we're comparing things to at idx 0
    sorted_distance_idxs = np.argsort(distance_matrix[idx])[:n_similar] # EX: [252, 102, 239, ...]
    # this is the index of the text that is most similar to the query (index 0)
    most_sim_idx = sorted_distance_idxs[1]

    # set the variable that will hold our matching FAQ
    faq_match = None

    for similar_idx in sorted_distance_idxs:
        # actual text data for display
        datum = data[similar_idx]

        # distance in cosine space from our text example to the similar text example
        distance = distance_matrix[idx][similar_idx]

        # how similar that text data is to our input example
        similarity =  1 - distance

        # set a confidence threshold
        # TODO
        if similar_idx == most_sim_idx and similarity >= 0.85:
                    faq_match = data[most_sim_idx]
        else:
            sorry = "Sorry, I'm not sure how to respond."


    # print the appropriate answer to the FAQ, or bring in a human to respond
    # TODO
    if faq_match is not None:
            return faqs[faq_match]
    else:
            return "Sorry, I'm not sure how to respond."
        
def input_question(question, data, feats):
    # TODO
    # Pass a question

    # add the user question and its vector representations to the corresponding lists, `data` and `feats`
    # insert them at index 0 so you know exactly where they are for later distance calculations
    if question is not None:
        data.insert(0, question)

    new_feats = indicoio.text_features(question)
    feats.insert(0, new_feats)

    return data, feats


def answer_question(test_questions_file):
    
    # reading files and creating dictionary of questions and answers
    training_questions_file = open("Questions.txt", "r", encoding = "utf-8")
    training_answers_file = open("Answers.txt", "r", encoding = "utf-8")
    
    training_questions = training_questions_file.readlines()
    training_answers = training_answers_file.readlines()
    
    faqs  = dict(zip(training_questions, training_answers))
    
    # TODO
    data = list(faqs.keys()) 

    feats = make_feats(data)

    testing_questions_file = open(test_questions_file, "r", encoding = "utf-8")
    testing_questions = testing_questions_file.readlines()

    results_file = open("qa_results.txt", "wb")

    for question in testing_questions:
        
        input_results = input_question(question, data, feats)
        new_data = input_results[0]
        new_feats = input_results[1]
        
        distance_matrix = calculate_distances(new_feats)
        
        idx = 0
        
        answer = similarity_text(idx, distance_matrix, new_data, faqs)
        results_file.write(answer.encode("utf-8"))


# In[ ]:


# Topic Modelling

df_topics = pd.DataFrame()
df_topics = pd.read_csv("Topics.txt", delimiter="\t ", engine="python",header=None, names=['topic'])


df_questions = pd.DataFrame()
df_questions = pd.read_csv("Questions.txt", engine="python", delimiter='\t',header=None, names=['questions'])


topic_model = pd.DataFrame()
topic_model = pd.concat([df_questions, df_topics], axis=1)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english", lowercase="True", strip_accents="ascii")

y = topic_model.topic
X = vectorizer.fit_transform(topic_model.questions.astype('U'))


# Importing the function for splitting data into test & train, 
# as well as F1 metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Splitting the data into 80% training, %20 for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)


# Importing the logistic regression function
from sklearn.linear_model import LogisticRegression

# Instantiate the classifier
log_reg = LogisticRegression(solver='lbfgs')

# The model will learn the relationship between the input 
# and the observation when fit is called on the data
log_reg.fit(X_train, y_train)

# Testing the model using the remaining test data
lr_predicted = log_reg.predict(X_test)


def topic_model(textfile):
    test_list = []
    infile = open(textfile, "r")
    
    outfile = open("topic_results.txt","w")
    for question in infile:
        test_list.append(question)
        
        processed = vectorizer.transform(test_list)
        
        result = log_reg.predict(processed)
        outfile.write(str(result[0]))
        outfile.write('\n')
        
        test_list = []
    infile.close()
    outfile.close()


# In[ ]:


if __name__ == "__main__":
    if(len(sys.argv) != 4):
    print("Check that you have all your arguments")
    
    else:
        if(sys.argv[1]=="topic"):
            topic_model(sys.argv[2])
            
        if(sys.argv[1]=="qa"):
            answer_question(sys.argv[2])

