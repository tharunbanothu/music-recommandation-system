#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd


# In[9]:


from typing import List, Dict


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[11]:


songs = pd.read_csv("C:/Users/banot/Downloads/songdata.csv")


# In[12]:


songs.head()


# In[13]:


songs = songs.sample(n=5000).drop('link', axis=1).reset_index(drop=True)


# In[14]:


songs['text'] = songs['text'].str.replace(r'\n', '')


# In[15]:


tfidf = TfidfVectorizer(analyzer='word', stop_words='english')


# In[16]:


lyrics_matrix = tfidf.fit_transform(songs['text'])


# In[17]:


cosine_similarities = cosine_similarity(lyrics_matrix) 


# In[18]:


similarities = {}


# In[21]:


for i in range(len(cosine_similarities)):
    # Now we'll sort each element in cosine_similarities and get the indexes of the songs. 
    similar_indices = cosine_similarities[i].argsort()[:-50:-1] 
    # After that, we'll store in similarities each name of the 50 most similar songs.
    # Except the first one that is the same song.
    similarities[songs['song'].iloc[i]] = [(cosine_similarities[i][x], songs['song'][x], songs['artist'][x]) for x in similar_indices][1:]


# In[22]:


class ContentBasedRecommender:
    def __init__(self, matrix):
        self.matrix_similar = matrix

    def _print_message(self, song, recom_song):
        rec_items = len(recom_song)
        
        print(f'The {rec_items} recommended songs for {song} are:')
        for i in range(rec_items):
            print(f"Number {i+1}:")
            print(f"{recom_song[i][1]} by {recom_song[i][2]} with {round(recom_song[i][0], 3)} similarity score") 
            print("--------------------")
        
    def recommend(self, recommendation):
        # Get song to find recommendations for
        song = recommendation['song']
        # Get number of songs to recommend
        number_songs = recommendation['number_songs']
        # Get the number of songs most similars from matrix similarities
        recom_song = self.matrix_similar[song][:number_songs]
        # print each item
        self._print_message(song=song, recom_song=recom_song)


# In[23]:


recommedations = ContentBasedRecommender(similarities)


# In[24]:


recommendation = {
    "song": songs['song'].iloc[10],
    "number_songs": 4 
}


# In[25]:


recommedations.recommend(recommendation)


# In[26]:


recommendation2 = {
    "song": songs['song'].iloc[120],
    "number_songs": 4 
}


# In[27]:


recommedations.recommend(recommendation2)


# In[ ]:




