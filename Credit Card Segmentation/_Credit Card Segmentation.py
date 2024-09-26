#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn import cluster


# In[6]:


df=pd.read_csv("credit.csv")


# In[7]:


df.head()


# In[8]:


df.isnull().sum()


# In[28]:


df.isnull().any()


# In[23]:


df['Income_Category'].value_counts()


# In[25]:


# Engineer categorical Variable with Ordianal number
Gender_map = {'M':2,
             'F': 1,}
Income_map = {'Less than $40K ':1,
             '$40K - $60K':2,
             '$60K - $80K':3,
             '$80K - $120K ':4,
             '$120K + ':5,
             'Unknown':6}


# In[26]:


df['Income_ordinal'] = df.Income_Category.map(Income_map)
df.head(20)


# In[19]:


df['Gender_ordinal'] = df.Gender.map(Gender_map)
df.head(20)


# In[29]:


df.dropna(subset = ['Income_ordinal'], inplace=True)


# In[31]:


clustering_data=df[['Gender_ordinal','Income_ordinal','Credit_Limit']]
from sklearn.preprocessing import MinMaxScaler
for i in clustering_data.columns:
    MinMaxScaler(i)
    
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
clusters=kmeans.fit_predict(clustering_data)
df['Credit_card_segments']=clusters


# In[33]:


df['Credit_card_segments']=df['Credit_card_segments'].map({0:'cluster 1',1:'cluster 2',2:'cluster 3',3:'cluster 4',4:'cluster 5'})
df['Credit_card_segments'].head()


# In[35]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Creating a 3D graph
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Loop through unique credit card segments and plot the scatter points
for segment in df['Credit_card_segments'].unique():
    # Extract data for each segment
    segment_data = df[df['Credit_card_segments'] == segment]
    ax.scatter(segment_data['Gender_ordinal'],
               segment_data['Income_ordinal'],
               segment_data['Credit_Limit'],
               s=40,  # Adjust marker size
               label=str(segment))  # Use segment as the label

# Adding axis labels and titles
ax.set_xlabel('Gender', fontsize=12, labelpad=10)
ax.set_ylabel('Income', fontsize=12, labelpad=10)
ax.set_zlabel('Credit Limit', fontsize=12, labelpad=10)
ax.set_title('3D Scatter Plot of Credit Card Segments', fontsize=15)

# Show legend
ax.legend()

# Show the plot
plt.show()


# In[38]:


import plotly.graph_objects as go
PLOT = go.Figure()
for i in list(df["Credit_card_segments"].unique()):
    

    PLOT.add_trace(go.Scatter3d(x = df[df["Credit_card_segments"]== i]['Gender_ordinal'],
                                y = df[df["Credit_card_segments"] == i]['Income_ordinal'],
                                z = df[df["Credit_card_segments"] == i]['Credit_Limit'],                        
                                mode = 'markers',marker_size = 6, marker_line_width = 1,
                                name = str(i)))
PLOT.update_traces(hovertemplate='Gender_ordinal: %{x} <br>Income_ordinal %{y} <br>Credit_Limit: %{z}')

    
PLOT.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                   scene = dict(xaxis=dict(title = 'Gender', titlefont_color = 'black'),
                                yaxis=dict(title = 'Income', titlefont_color = 'black'),
                                zaxis=dict(title = 'CREDIT_LIMIT', titlefont_color = 'black')),
                   font = dict(family = "Gilroy", color  = 'black', size = 12))


# In[ ]:




