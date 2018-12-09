import math
import os
from random import sample
import pickle as pickle
from tqdm import tqdm
from scipy.spatial.distance import cdist
import json
import numpy as np
import indicoio
from texttable import Texttable
indicoio.config.api_key = "edd63dd43797ccb9cb13606856193622"

'''
Use indico's Text Features API to find text similarity and create a customer support bot that automatically responds to FAQs from users.
Tutorial: https://indico.io/blog/faqs-bot-text-features-api/
'''

# faqs = {
#     'Where can I find my API Key?':'Hi there! You receive an API key upon sign up. After you confirm your email you will be able to log in to your dashboard at indico.io/dashboard and see your API key on the top of the screen.',
#     'Can indico be downloaded as a package and used offline?':'Unfortunately, no. However we do have a paid option for on premise deployment for enterprise clients.',
#     'What is indico API credit?':'Hello! indico API credit is what we use to keep track of usage. If you send in 100 bits of text into our API you are charged 100 credits, essentially one credit is consumed per datapoint analyzed. Every user gets 10,000 free API credits per month.',
#     'Would I be able to set up a Pay as You Go account and have it stop if I reach 10,000 calls so that I won\'t be charged if I accidentally go over the limit?':'Hi there! Yep, the best way for you to do this would be to sign up for a pay as you go account and don\'t put in a credit card (we don\'t require you to). When you hit 10,000 you will be locked out of your account and unable to make more calls until you put a credit card in or you can wait until the first of the month when it resets to 10,000.',
#     'Hello! When I try to install indico with pip, I get this error on Windows. Do you know why?':'Hello, please try following the steps listed here: https://indico.io/blog/getting-started-indico-tutorial-for-beginning-programmers/#windows and let us know if you still continue to have problems.'
# }

def make_feats(data):
    """
    Send our text data throught the indico API and return each text example's text vector representation
    """
    # TODO
    chunks = [data[x:x+100] for x in range(0, len(data), 100)]
    feats = []

    # just a progress bar to show us how much we have left
    for chunk in tqdm(chunks):
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
    t = Texttable()
    t.set_cols_width([50, 20])

    # these are the indexes of the texts that are most similar to the text at data[idx]
    # note that this list of 10 elements contains the index of text that we're comparing things to at idx 0
    sorted_distance_idxs = np.argsort(distance_matrix[idx])[:n_similar] # EX: [252, 102, 239, ...]
    # this is the index of the text that is most similar to the query (index 0)
    most_sim_idx = sorted_distance_idxs[1]

    # header for texttable
    t.add_rows([['Text', 'Similarity']])
    print(t.draw())

    # set the variable that will hold our matching FAQ
    faq_match = None

    for similar_idx in sorted_distance_idxs:
        # actual text data for display
        datum = data[similar_idx]

        # distance in cosine space from our text example to the similar text example
        distance = distance_matrix[idx][similar_idx]

        # how similar that text data is to our input example
        similarity =  1 - distance

        # add the text + the floating point similarity value to our Texttable() object for display
        t.add_rows([[datum, str(round(similarity, 2))]])
        print(t.draw())

        # set a confidence threshold
        # TODO
        if similar_idx == most_sim_idx and similarity >= 0.75:
                    faq_match = data[most_sim_idx]
        else:
            sorry = "Sorry, I'm not sure how to respond. Let me find someone who can help you."


    # print the appropriate answer to the FAQ, or bring in a human to respond
    # TODO
    if faq_match is not None:
            print ("A: %r" % faqs[faq_match])
    else:
            print ("sorry")

def input_question(data, feats):
    # TODO
    # input a question
    question = input("What is your question? ")

    # add the user question and its vector representations to the corresponding lists, `data` and `feats`
    # insert them at index 0 so you know exactly where they are for later distance calculations
    if question is not None:
        data.insert(0, question)

    new_feats = indicoio.text_features(question)
    feats.insert(0, new_feats)

    return data, feats

def run():

    # reading files and creating dictionary of questions and answers
    questions_file = open("Questions.txt", "r", encoding = "utf-8")
    answers_file = open("Answers.txt", "r", encoding = "utf-8")

    input_questions = questions_file.readlines()
    input_answers = answers_file.readlines()

    faqs  = dict(zip(input_questions, input_answers))


    # TODO
    data = list(faqs.keys())
    print ("FAQ data received. Finding features.")

    feats = make_feats(data)

    with open('faq_feats.pkl', 'wb') as f:
        pickle.dump(feats, f)
    print ("FAQ features found!")

    with open('faq_feats.pkl', 'rb') as f:
        feats = pickle.load(f)
    print ("Features found -- success! Calculating similarities...")

    input_results = input_question(data, feats)
    new_data = input_results[0]
    new_feats = input_results[1]

    distance_matrix = calculate_distances(new_feats)
    print ("Similarities found. Generating table.")

    idx = 0
    similarity_text(idx, distance_matrix, new_data, faqs)
    print ('\n' + '-' * 80)

if __name__ == "__main__":
    run()