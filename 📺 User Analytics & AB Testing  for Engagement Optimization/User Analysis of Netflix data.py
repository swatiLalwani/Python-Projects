#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries


# In[60]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import scipy.stats


# In[61]:


#Read the data


# In[62]:


df=pd.read_csv("Netflix Userbase.csv")


# In[63]:


df.head()


# In[64]:


#Checking if any duplicates with user id
df.duplicated(subset='User ID').sum()


# In[65]:


#Dropping unwanted columns
df.drop(['Plan Duration','User ID'],axis=1,inplace=True)


# In[66]:


df.shape


# In[67]:


df.info()


# In[68]:


df.describe()


# In[69]:


#check if the categorical variables have appropriate number of levels
df_category=df[['Subscription Type','Country','Gender','Device']]
df_category.nunique()


# In[70]:


#check if the categorical variables have appropriate levels
for i in df_category.columns:
    print(i.upper(),":",df_category[i].unique())


# In[71]:


#Univariate Analysis


# In[72]:


var='Subscription Type'

plt.figure(figsize = (6,4))
#count plot
plt.subplot(1,2,1)
sns.countplot(x=var,data=df_category)
plt.title(f'Count Plot - {var}')

#piechart
plt.subplot(1,2,2)
counts=df_category[var].value_counts()
plt.pie(counts,labels=counts.index,autopct="%0.2f%%")
plt.title(f'Pie Chart - {var}')

#Adjust layout
plt.tight_layout()

#show the plots
plt.show()


# We see that there are subscriptions of Basic type to Netflix more than ofStandard and Premium types.
# We can see that Standard and Premium types are nearly the same.
# However, after all, the data is not that unbalanced when it comes to Subscription Type attribute.

# In[26]:


var='Country'

plt.figure(figsize = (18,8))
#count plot
plt.subplot(1,2,1)
sns.countplot(x=var,data=df_category)
plt.title(f'Count Plot - {var}')

#piechart
plt.subplot(1,2,2)
counts=df_category[var].value_counts()
plt.pie(counts,labels=counts.index,autopct="%0.2f%%")
plt.title(f'Pie Chart - {var}')

#Adjust layout
plt.tight_layout()

#show the plots
plt.show()


# We notice that United States and Canada capture most of Netflix users origins with 18% for each, followed by United Kingdom with 12.7% of users' origins. Then come the rest of countries (Australia, Germany, France, Brazil, Mexico, Spain, Italy), all with the same percentage of 7.32%.
# 
# 

# In[27]:


var='Gender'

plt.figure(figsize = (6,4))
#count plot
plt.subplot(1,2,1)
sns.countplot(x=var,data=df_category)
plt.title(f'Count Plot - {var}')

#piechart
plt.subplot(1,2,2)
counts=df_category[var].value_counts()
plt.pie(counts,labels=counts.index,autopct="%0.2f%%")
plt.title(f'Pie Chart - {var}')

#Adjust layout
plt.tight_layout()

#show the plots
plt.show()


# The data collected based on Gender is satisfiably balanced, Netflix users are 50% men as it is for women.

# In[29]:


var='Device'

plt.figure(figsize = (10,4))
#count plot
plt.subplot(1,2,1)
sns.countplot(x=var,data=df_category)
plt.title(f'Count Plot - {var}')

#piechart
plt.subplot(1,2,2)
counts=df_category[var].value_counts()
plt.pie(counts,labels=counts.index,autopct="%0.2f%%")
plt.title(f'Pie Chart - {var}')

#Adjust layout
plt.tight_layout()

#show the plots
plt.show()


# As the Gender column, Device column is so balanced, 25% of Netflix users watch Netlix using Smartphones. The same percentage of users goes to Tablet, Smart TV and Laptop.

# In[31]:


var='Monthly Revenue'

plt.figure(figsize = (18,8))
#count plot
plt.subplot(1,2,1)
sns.countplot(x=var,data=df)
plt.title(f'Count Plot - {var}')

#piechart
plt.subplot(1,2,2)
sns.boxplot(y=var,data=df)
plt.title(f'Boxplot - {var}')

#Adjust layout
plt.tight_layout()

#show the plots
plt.show()


# We can see that Netflix users are nearly equally devided when it comes to the monthly revenues Netflix is profitting from them. However, more users are benifitting Netflix by 12 USD per moth more than other revenues.

# In[32]:


#descriptive stats of Monthly Revenue
df['Monthly Revenue'].describe()


# In[37]:


var='Monthly Revenue'

plt.figure(figsize = (18,8))
#count plot
plt.subplot(1,2,1)
#Filtering data for a value close to 75 th percentile only for better visualization
sns.histplot(x=var,data=df[df['Monthly Revenue']<14])
plt.title(f'Histogram - {var}')

#piechart
plt.subplot(1,2,2)
sns.boxplot(y=var,data=df[df['Monthly Revenue']<14])
plt.title(f'Boxplot - {var}')

#Adjust layout
plt.tight_layout()

#show the plots
plt.show()


# In[38]:


##Bivariate Analysis


# In[39]:


df.columns


# In[40]:


ct_conversion_Subscription_type=pd.crosstab(df['Subscription Type'],df['Monthly Revenue'],normalize='index')
ct_conversion_Subscription_type


# In[42]:


ct_conversion_Subscription_type.plot.bar(stacked=True);


# In[49]:


ct_conversion_Age=pd.crosstab(df['Age'],df['Monthly Revenue'],normalize='index')
print(ct_conversion_Age.sort_values(by='Age', ascending=False))
ct_conversion_Age.plot.bar(stacked=True);


# In[50]:


ct_conversion_Country=pd.crosstab(df['Country'],df['Monthly Revenue'],normalize='index')
print(ct_conversion_Country.sort_values(by='Country', ascending=False))
ct_conversion_Country.plot.bar(stacked=True);


# In[52]:


ct_conversion_Gender=pd.crosstab(df['Gender'],df['Monthly Revenue'],normalize='index')
print(ct_conversion_Gender)
ct_conversion_Gender.plot.bar(stacked=True);


# In[54]:


sns.boxplot(x='Monthly Revenue',y='Subscription Type', data = df)


# In[56]:


sns.boxplot(x='Monthly Revenue',y='Age', data = df[df['Age']<30])


# In[80]:


from scipy.stats import chi2_contingency
alpha=0.05
for var in df_category.columns:
    if var!= 'Monthly Revenue':
        #Create a contigency table (cross-tabulation)
        contigency_table=pd.crosstab(df_category[var],df['Monthly Revenue'])
        
        #perform chi-squared test
        chi2, p, _, _=chi2_contingency(contigency_table)
        
        #Display the results
        print(f'\nChi-squared test for {var} vs. Monthly Revenue:')
        print(f'Chi-squared value:{chi2}')
        print(f'p-value:{p}')
        
        #check for significance
        if p < alpha:
            print(f'The difference in profit across {var} is statistically significant.')
        else:
            print(f'There is no significant difference in the profit accross {var}.')


# In[89]:


from scipy.stats import shapiro, levene ,  mannwhitneyu

# Check the number of data points in each group
true_group_size = len(df[df['Monthly Revenue'] == True]['Age'])
false_group_size = len(df[df['Monthly Revenue'] == False]['Age'])

# Normality Assumption
if true_group_size >= 3:
    shapiro_stat_true, shapiro_p_value_true = shapiro(df[df['Monthly Revenue'] == True]['Age'])
    print(f'Shapiro-Wilk test for normality (True group): p-value = {shapiro_p_value_true}')
else:
    print(f"True group has fewer than 3 data points (size: {true_group_size}), skipping Shapiro-Wilk test.")

if false_group_size >= 3:
    shapiro_stat_false, shapiro_p_value_false = shapiro(df[df['Monthly Revenue'] == False]['Age'])
    print(f'Shapiro-Wilk test for normality (False group): p-value = {shapiro_p_value_false}')
else:
    print(f"False group has fewer than 3 data points (size: {false_group_size}), skipping Shapiro-Wilk test.")

# Equality of variances assumption
if true_group_size >= 3 and false_group_size >= 3:
    levene_stat, levene_p_value = levene(df[df['Monthly Revenue'] == True]['Age'], df[df['Monthly Revenue'] == False]['Age'])
    print(f"Levene's test for equality of variances: p-value = {levene_p_value}")
else:
    print("Levene's test skipped due to insufficient data points in one or both groups.")


# In[ ]:




