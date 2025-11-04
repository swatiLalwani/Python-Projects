#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd


# In[53]:


items=pd.read_csv('WMT_Grocery.csv')


# # EXPLORATORY DATA ANALYSIS

# In[54]:


items.head()


# In[55]:


print(items.shape)


# In[88]:


print(items['PRODUCT_NAME'])


# In[57]:


items.isnull().sum()


# In[58]:


items.duplicated().sum()


# # POPULARITY BASED RECOMMENDED SYSTEM

# In[59]:


popular = items[items['PRICE_RETAIL']<=50].sort_values('CATEGORY',ascending=False).head(5)


# In[60]:


popular = popular.assign(PRODUCT_IMAGE=['https://i5.walmartimages.com/seo/Chobani-Non-Fat-Greek-Yogurt-Black-Cherry-on-the-Bottom-5-3-oz-4-Count-Plastic_515b3666-5a7d-435d-a30a-6bb494f7d876.bd706e61152e1210a2d6b9238e24aa2c.jpeg?odnHeight=2000&odnWidth=2000&odnBg=FFFFFF', 'https://i5.walmartimages.com/seo/Stonyfield-Organic-Greek-Nonfat-Yogurt-Plain-32-oz_1bd760cb-3b3b-4a7c-84ca-fe74628e44a0.c2f0e0dd7341c4189ad017dd6849ec50.jpeg?odnHeight=2000&odnWidth=2000&odnBg=FFFFFF','https://i5.walmartimages.com/seo/Great-Value-Original-Strawberry-Banana-Lowfat-Yogurt-6-oz-Cups-4-Count_32beea4e-78f2-4048-b653-109fb4fbba44.71041f59ab7d00cca1296119bdcf4b3d.jpeg?odnHeight=2000&odnWidth=2000&odnBg=FFFFFF','https://i5.walmartimages.com/seo/Yoplait-Original-Strawberry-Cheesecake-Low-Fat-Yogurt-6-oz-Yogurt-Cup_8c5f25ff-b77f-4a6f-bf41-30b66b53d9cc.17d9e658059c7825104cc87e0c23ecf1.jpeg?odnHeight=2000&odnWidth=2000&odnBg=FFFFFF','https://i5.walmartimages.com/seo/Yoplait-Original-Low-Fat-Yogurt-Variety-Pack-8-Yogurt-Cups-48-oz_432fdefc-f535-426d-b61d-41cf5bc85ea5.0ca802f9e31e768e89a827f96246346f.jpeg?odnHeight=2000&odnWidth=2000&odnBg=FFFFFF'])


# In[61]:


popular.head(5)


# In[62]:


popular= popular[['CATEGORY','PRODUCT_NAME','PRODUCT_URL','PRICE_RETAIL','PRODUCT_SIZE','PRODUCT_IMAGE']]


# In[63]:


popular


# In[64]:


popular['PRODUCT_IMAGE
        '][4688]


# # COLLABORATIVE RECOMMENDED SYSTEM

# In[65]:


x = items.groupby('CATEGORY').count()['PRICE_RETAIL'] >= 1
product = x[x].index


# In[66]:


filtered_items= items[items['CATEGORY'].isin(product)]


# In[67]:


y = filtered_items.groupby('PRODUCT_NAME').count()['PRICE_RETAIL']>= 1
famous_products = y[y].index


# In[68]:


final_items = filtered_items[filtered_items['PRODUCT_NAME'].isin(famous_products)]


# In[69]:


pt = final_items.pivot_table(index='PRODUCT_NAME',columns='CATEGORY',values='PRICE_RETAIL')


# In[70]:


pt.fillna(0,inplace=True)


# In[71]:


pt


# In[72]:


from sklearn.metrics.pairwise import cosine_similarity


# In[73]:


similarity_scores = cosine_similarity(pt)


# In[74]:


similarity_scores.shape


# In[83]:


def recommend(product_name):
    # index fetch
    index = np.where(pt.index== product_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:11]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = items[items['PRODUCT_NAME'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('PRODUCT_NAME')['PRODUCT_NAME'].values))
        item.extend(list(temp_df.drop_duplicates('PRODUCT_NAME')['CATEGORY'].values))
        item.extend(list(temp_df.drop_duplicates('PRODUCT_NAME')['PRICE_RETAIL'].values))
        item.extend(list(temp_df.drop_duplicates('PRODUCT_NAME')['PRODUCT_URL'].values))
        data.append(item)
    
    return data


# In[84]:


recommend('Farm Fresh Peach Moscato 750 Ml')


# In[85]:


pt.index[20000]


# In[86]:


import pickle


# In[81]:


pickle.dump(popular,open('popular1.pkl','wb'))


# In[87]:


pickle.dump(pt,open('pt.pkl','wb'))
pickle.dump(items,open('items.pkl','wb'))
pickle.dump(similarity_scores,open('similarity_scores.pkl','wb'))


# In[ ]:




