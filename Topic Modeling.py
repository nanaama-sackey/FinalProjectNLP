#!/usr/bin/env python
# coding: utf-8

# In[58]:
import sys

# Initialising dataframes in pandas containing the topics and questions
import pandas as pd

df_topics = pd.DataFrame()
df_topics = pd.read_csv("Placebo/Topics.txt", delimiter="\t ", engine="python",header=None, names=['topic'])


# In[59]:


df_questions = pd.DataFrame()
df_questions = pd.read_csv("Placebo/Questions.txt", engine="python", delimiter='\t',header=None, names=['questions'])


# In[60]:


topic_model = pd.DataFrame()
topic_model = pd.concat([df_questions, df_topics], axis=1)


# In[61]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english", lowercase="True", strip_accents="ascii")

y = topic_model.topic
X = vectorizer.fit_transform(topic_model.questions.astype('U'))


# In[62]:


# Importing the function for splitting data into test & train, 
# as well as F1 metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Splitting the data into 80% training, %20 for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)


# In[63]:


# Importing the logistic regression function
from sklearn.linear_model import LogisticRegression

# Instantiate the classifier
log_reg = LogisticRegression(solver='lbfgs')

# The model will learn the relationship between the input 
# and the observation when fit is called on the data
log_reg.fit(X_train, y_train)

# Testing the model using the remaining test data
lr_predicted = log_reg.predict(X_test)


# In[64]:


# Evaluating the F1 measure of the logistic regression model
f1_score(y_test, lr_predicted, average='weighted') 


# In[65]:


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


# In[66]:


topic_model("/Users/lvz/Documents/Code/Python/FinalProjectNLP/Placebo/tester.txt")


# In[67]:


if(len(sys.argv) != 4):
    print("Check that you have all your arguments")

else:
    if(sys.argv[1]=="topic"):
        topic_model(sys.argv[2])


# In[ ]:




