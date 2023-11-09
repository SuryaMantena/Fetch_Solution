#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import spacy
import string
import gensim
import operator
import re


# In[2]:


# Load data from a CSV file
df = pd.read_csv('FINAL.csv')
df.head(5)


# In[3]:


df.shape


# In[4]:


def concatenate_unique_values(row):
    # Convert all values to strings and remove duplicates by converting to a set.
    unique_values = set(row.dropna().astype(str))
    # Sort the values to maintain consistent order.
    sorted_unique_values = sorted(unique_values)
    # Join the unique values into a string with space as separator.
    return ' '.join(sorted_unique_values)

# Apply the function to each row of the DataFrame.
df['OFFER_new'] = df.apply(concatenate_unique_values, axis=1)

# Display the DataFrame with the new 'OFFER' column
print(df['OFFER_new'])


# In[5]:


output_file = 'offfer_retailer_nonascii.csv'
df.to_csv(output_file, index=False)


# In[6]:


df = df.drop_duplicates()
print(df)


# In[7]:


from spacy.lang.en.stop_words import STOP_WORDS

spacy_nlp = spacy.load('en_core_web_sm')

#create list of punctuations and stopwords
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS

#function for data cleaning and processing
#This can be further enhanced by adding / removing reg-exps as desired.





def spacy_tokenizer(sentence):
 
    #remove distracting single quotes
    sentence = re.sub('\'','',sentence)

    #remove digits adnd words containing digits
    sentence = re.sub('\w*\d\w*','',sentence)

    #replace extra spaces with single space
    sentence = re.sub(' +',' ',sentence)

    
    #remove non-breaking new line characters
    sentence = re.sub(r'\n',' ',sentence)
    
    #remove punctunations
    sentence = re.sub(r'[^\w\s]',' ',sentence)
    
    #creating token object
    tokens = spacy_nlp(sentence)
    
    #lower, strip and lemmatize
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
    
    
    #return tokens
    return tokens


# In[8]:


print ('Cleaning and Tokenizing...')
df['OFFER_new_tokenized'] = df['OFFER_new'].map(lambda x: spacy_tokenizer(x))

df.head()


# In[9]:


OFFER_df= df['OFFER_new_tokenized']
OFFER_df.head()


# In[10]:


from gensim import corpora

#creating term dictionary
get_ipython().run_line_magic('time', 'dictionary = corpora.Dictionary(OFFER_df)')

#filter out terms which occurs in less than 4 documents and more than 20% of the documents.
#NOTE: Since we have smaller dataset, we will keep this commented for now.

#dictionary.filter_extremes(no_below=4, no_above=0.2)

#list of few which which can be further removed
stoplist = set('hello and if this can would should could tell ask stop come go')
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_ids)


# In[11]:


#print top 50 items from the dictionary with their unique token-id
dict_tokens = [[[dictionary[key], dictionary.token2id[dictionary[key]]] for key, value in dictionary.items() if key <= 50]]
print (dict_tokens)


# In[12]:


corpus = [dictionary.doc2bow(desc) for desc in OFFER_df]

word_frequencies = [[(dictionary[id], frequency) for id, frequency in line] for line in corpus[0:3]]

print(word_frequencies)


# In[13]:


OFFER_tfidf_model = gensim.models.TfidfModel(corpus, id2word=dictionary)
OFFER_lsi_model = gensim.models.LsiModel(OFFER_tfidf_model[corpus], id2word=dictionary, num_topics=300)


# In[14]:


gensim.corpora.MmCorpus.serialize('OFFER_tfidf_model_mm', OFFER_tfidf_model[corpus])
gensim.corpora.MmCorpus.serialize('OFFER_lsi_model_mm',OFFER_lsi_model[OFFER_tfidf_model[corpus]])


# In[15]:


#Load the indexed corpus
OFFER_tfidf_corpus = gensim.corpora.MmCorpus('OFFER_tfidf_model_mm')
OFFER_lsi_corpus = gensim.corpora.MmCorpus('OFFER_lsi_model_mm')

print(OFFER_tfidf_corpus)
print(OFFER_lsi_corpus)


# In[16]:


from gensim.similarities import MatrixSimilarity

OFFER_index = MatrixSimilarity(OFFER_lsi_corpus, num_features = OFFER_lsi_corpus.num_terms)


# In[17]:


from operator import itemgetter

def search_similar_OFFER(search_term):

    query_bow = dictionary.doc2bow(spacy_tokenizer(search_term))
    query_tfidf = OFFER_tfidf_model[query_bow]
    query_lsi = OFFER_lsi_model[query_tfidf]

    OFFER_index.num_best = 5

    OFFER_list = OFFER_index[query_lsi]

    OFFER_list.sort(key=itemgetter(1), reverse=True)
    OFFER_names = []

    for j, OFFER in enumerate(OFFER_list):

        OFFER_names.append (
            {
                'Relevance': round((OFFER[1] * 100),2),
                'OFFER': df['OFFER'][OFFER[0]],
                'Retailer': df['RETAILER'][OFFER[0]]
                
            }

        )
        if j == (OFFER_index.num_best-1):
            break

    return pd.DataFrame(OFFER_names, columns=['Relevance','OFFER','Retailer'])


# In[31]:


y = str(input("Enter your query: "))
search_similar_OFFER(y)


# In[ ]:





# In[ ]:





# In[ ]:




