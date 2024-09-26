#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[68]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[69]:


movies.head()


# In[70]:


credits.head()


# In[71]:


print(movies.shape)


# In[72]:


print(credits.shape)


# In[79]:


movies=movies.merge(credits,on='title')


# In[80]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[81]:


movies.head()


# In[94]:


movies.isnull().sum()


# In[95]:


movies.dropna(inplace=True)


# In[96]:


movies.iloc[0].genres


# In[57]:


import ast


# In[58]:


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[60]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[61]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[62]:


ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[63]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[64]:


movies['cast'] = movies['cast'].apply(convert)
movies.head()


# In[97]:


movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[98]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[99]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[100]:


#movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(5)


# In[101]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[102]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[103]:


movies.head()


# In[104]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[ ]:





# In[ ]:





# In[ ]:





# In[105]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[106]:


new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()


# In[107]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()


# In[108]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[109]:


vector = cv.fit_transform(new['tags']).toarray()


# In[110]:


vector.shape


# In[111]:


from sklearn.metrics.pairwise import cosine_similarity


# In[112]:


similarity = cosine_similarity(vector)


# In[113]:


similarity


# In[114]:


new[new['title'] == 'The Lego Movie'].index[0]


# In[115]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)


# In[116]:


recommend('Avatar')


# In[ ]:




