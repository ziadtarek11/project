#!/usr/bin/env python
# coding: utf-8

# <center> <h1  style="background-color:#7CCA62; color:white;"> Acquire and Prepare Data </h1></center>

# <h3 style="background-color:#E5F4E0; color:#387026;">  Importing the libraries </h3>

# In[1]:


import numpy as np
import pandas as pd


# <h3 style="background-color:#E5F4E0; color:#387026;">  Importing the dataset </h3>

# In[2]:


#NBA players
df=pd.read_csv("Playerdata.csv")
df


# In[3]:


df.tail(10)


# In[4]:


df.describe()


# In[5]:


df.head(10)#try df.head(number)


# <h3 style="background-color:#E5F4E0; color:#387026;">  Check missing Values </h3>
#   

# In[6]:


with pd.option_context('display.max_rows', 457):
    print(df.isna())


# In[7]:


round(df['Salary'].mean())


# 
# 
# <h3 style="background-color:#E5F4E0; color:#387026;">  Explain what you found? </h3>
#   

# # i found some missing values in two columns ("Salary","College")

# <h3 style="background-color:#E5F4E0; color:#387026;">  Write code for handeling any missing  </h3>
#   

# In[8]:


df['Salary'].fillna(round(df['Salary'].mean()),inplace=True)


# In[9]:


with pd.option_context('display.max_rows', 457, 'display.max_columns' , 7):
    print(df.isna())


# In[10]:


df['College'].fillna('LSU',inplace=True)
df.isna()


# In[11]:


df.info()
df['Salary']


# <h3 style="background-color:#E5F4E0; color:#387026;">  you see that Age & Weight & Salary not accurate data types , write your code to handle them </h3>
#   

# In[12]:


df['Age'] = df['Age'].str.extract('(\d+)').astype(float)


# In[13]:


df['Age'].isna()
condition = df["Age"] > 37
df.loc[condition, 'Age'] = df.loc[condition, 'Age'].mean()


# In[15]:





# In[ ]:


df['Age']


# In[ ]:


condition_weight = df["Salary"] > 9000000
df.loc[condition_weight, 'Salary'] = df.loc[condition, 'Salary'].mean()
df["Salary"]


# In[ ]:


condition_weight = df["Weight"] > 250
df.loc[condition_weight, 'Weight'] = df.loc[condition, 'Weight'].mean()


# In[ ]:


round(df["Weight"])


# <h3 style="background-color:#E5F4E0; color:#387026;"> Display Age and Salary as a dataframe</h3>
# 

# In[ ]:


pd.DataFrame(df,columns=['Age','Salary'])


# <h3 style="background-color:#E5F4E0; color:#387026;">  Filter data according to Salary below 60000</h3> 

# In[ ]:


filt_salary= (df['Salary']<60000)
df[filt_salary]


# <h3 style="background-color:#E5F4E0; color:#387026;">  Remove the rows that satisfy the condition </h3>

# In[ ]:


condition = df['Salary'] < 60000
df=df.drop(df[condition].index)
print(df)


# In[ ]:





# <h3 style="background-color:#E5F4E0; color:#387026;">  Encoding categorical data - Example</h3> 

# In[ ]:


df['Position'].unique()


# In[ ]:


#example categorical - ordinal (with order)
#PG: 1 , SG:2, SF:3, PF: 4 , C:5
ser1=pd.Series([1,2,3,4,5],index=['PG','SG','SF','PF','C'])
df['Position']=df['Position'].replace(ser1)


# In[ ]:




