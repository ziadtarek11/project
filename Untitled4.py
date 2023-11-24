#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import metrics
df = pd.read_csv("laliga.csv")
print(df.to_string())


# In[4]:


df.info()
df.describe()


# In[5]:


print(df.shape) # number of rows and columns
print(df.size) # total number of elements
print(df.dtypes) # data types of each column


# In[6]:


df.isnull().sum() # check for missing values


# In[7]:


df.mean() # mean of each column


# In[8]:


df.median() # median of each column
df.var() # variance of each column
df.corr() # correlation between columns
df.cov() # covariance between columns


# In[10]:


player_name = set(df["Player Name"])
print(f"There are {len(player_name)} player name.\n")


# In[11]:


x = df["Goals"].median()
df["Goals"].fillna(x, inplace = True)

y = df["Assist"].median()
df["Assist"].fillna(y, inplace = True)

z = df["YC"].median()
df["YC"].fillna(z, inplace = True)

v = df["RC"].median()
df["RC"].fillna(v, inplace = True)

e = df["SPG"].median()
df["SPG"].fillna(e, inplace = True)

h = df["AW"].median()
df["AW"].fillna(h, inplace = True)

d = df["MOTM"].median()
df["MOTM"].fillna(d, inplace = True)
df


# In[12]:


df.isna().any(axis=0)


# In[13]:


df_top10 = df.sort_values(by='Rating', ascending=False).head(10)

# Create a bar chart for each column in the DataFrame
plt.figure(figsize=(20, 20))
df_top10.plot(kind='bar', subplots=True, sharex=False, sharey=False, layout=(4, 3), figsize=(20, 20))

# Set the title and axis labels for each chart
for i, ax in enumerate(plt.gcf().axes):
    ax.set_title(df_top10.columns[i])
    ax.set_xlabel('Player Name')
    ax.set_ylabel('Value')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)

# Show the plot
plt.show()


# In[16]:


df_top10 = df.sort_values(by='Rating', ascending=False).head(10)

# Create a scatter plot for each desired column
plt.figure(figsize=(12, 8))
plt.scatter(df_top10['Player Name'], df_top10['Rating'], color='red')
plt.scatter(df_top10['Player Name'], df_top10['Goals'], color='blue')
plt.scatter(df_top10['Player Name'], df_top10['Assist'], color='green')
plt.scatter(df_top10['Player Name'], df_top10['YC'], color='yellow')
plt.scatter(df_top10['Player Name'], df_top10['RC'], color='black')
plt.scatter(df_top10['Player Name'], df_top10['SPG'], color='purple')
plt.scatter(df_top10['Player Name'], df_top10['MOTM'], color='orange')
plt.scatter(df_top10['Player Name'], df_top10['AW'], color='pink')

# Set the title and axis labels
plt.title('Top 10 Players Scatter Plot')
plt.xlabel('Player Name')
plt.ylabel('Value')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)

# Show the plot
plt.show()


# In[17]:


df.hist(figsize=(10,10))


# In[23]:


print(" The min and max player's age")
print(f'Minimum Age: {min(df["Age"])}')
print(f'Maxmum Age: {max(df["Age"])}')


# In[22]:


print(" The min and max MinP")
print(f'Minimum Minp: {min(df["MinP"])}')
print(f'Maxmum Minp: {max(df["MinP"])}')


# In[24]:


df_sorted = df.sort_values('Goals', ascending=False)

# Get the top 10 players by number of goals
top_10 = df_sorted.head(10)

# Create a count plot of the top 10 players by number of goals
sns.set(style='whitegrid')
plt.figure(figsize=(10, 5))
ax = sns.countplot(x='Player Name', data=top_10)
ax.set(title='Top 10 Players by Number of Goals', xlabel='Player Name', ylabel='Count')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[25]:


numerical_cols = ['Age', 'App', 'MinP', 'Goals', 'Assist', 'YC', 'RC', 'SPG', 'PS%', 'AW', 'MOTM', 'Rating']
df_numerical = df[numerical_cols]

# Calculate the correlation matrix
corr_matrix = df_numerical.corr()

# Create a heatmap of the correlation matrix
sns.set(style='white')
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[110]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import metrics

# Load the dataset
df = pd.read_csv('laliga.csv')


# Convert non-numeric columns to numeric
le = LabelEncoder()
df['Team-name'] = le.fit_transform(df['Team-name'])
lt = LabelEncoder()
df['Position'] = lt.fit_transform(df['Position'])
lp = LabelEncoder()
df['App'] = lp.fit_transform(df['App'])
lc = LabelEncoder()
df['PS%'] = lc.fit_transform(df['PS%'])
lh = LabelEncoder()
df['Player Name'] = lh.fit_transform(df['Player Name'])


# fill in missing values with column means
df.fillna(df.mean(), inplace=True)

# scale the data using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)


# Drop non-numeric columns
X = df.drop(['Player Name', 'Rating'], axis=1)
y = df['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Decision Tree model
Dtree = DecisionTreeRegressor()
Dtree.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = Dtree.predict(X_test)

# Evaluate the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[111]:


from sklearn.neighbors import KNeighborsRegressor
le = LabelEncoder()
df['Team-name'] = le.fit_transform(df['Team-name'])
lt = LabelEncoder()
df['Position'] = lt.fit_transform(df['Position'])
lp = LabelEncoder()
df['App'] = lp.fit_transform(df['App'])
lc = LabelEncoder()
df['PS%'] = lc.fit_transform(df['PS%'])
lh = LabelEncoder()
df['Player Name'] = lh.fit_transform(df['Player Name'])


# fill in missing values with column means
df.fillna(df.mean(), inplace=True)
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
y_pred1 = knn.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred1))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))


# In[112]:


from sklearn.neighbors import KNeighborsRegressor
le = LabelEncoder()
df['Team-name'] = le.fit_transform(df['Team-name'])
lt = LabelEncoder()
df['Position'] = lt.fit_transform(df['Position'])
lp = LabelEncoder()
df['App'] = lp.fit_transform(df['App'])
lc = LabelEncoder()
df['PS%'] = lc.fit_transform(df['PS%'])
lh = LabelEncoder()
df['Player Name'] = lh.fit_transform(df['Player Name'])


# fill in missing values with column means
df.fillna(df.mean(), inplace=True)
MLP= MLPRegressor()
MLP.fit(X_train, y_train)
y_pred2 = MLP.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred1))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))


# In[113]:


from sklearn.neighbors import KNeighborsRegressor
le = LabelEncoder()
df['Team-name'] = le.fit_transform(df['Team-name'])
lt = LabelEncoder()
df['Position'] = lt.fit_transform(df['Position'])
lp = LabelEncoder()
df['App'] = lp.fit_transform(df['App'])
lc = LabelEncoder()
df['PS%'] = lc.fit_transform(df['PS%'])
lh = LabelEncoder()
df['Player Name'] = lh.fit_transform(df['Player Name'])


# fill in missing values with column means
df.fillna(df.mean(), inplace=True)
Dtree= DecisionTreeRegressor()
Dtree.fit(X_train, y_train)
y_pred3 = Dtree.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred1))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))


# In[114]:


from sklearn.neighbors import KNeighborsRegressor
le = LabelEncoder()
df['Team-name'] = le.fit_transform(df['Team-name'])
lt = LabelEncoder()
df['Position'] = lt.fit_transform(df['Position'])
lp = LabelEncoder()
df['App'] = lp.fit_transform(df['App'])
lc = LabelEncoder()
df['PS%'] = lc.fit_transform(df['PS%'])
lh = LabelEncoder()
df['Player Name'] = lh.fit_transform(df['Player Name'])


# fill in missing values with column means
df.fillna(df.mean(), inplace=True)
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
#To do so, we will use our test data and see how accurately our algorithm predicts the percentage score
y_pred = regressor.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[116]:


print(y_test.shape) 
print(y_pred.shape) 


# In[120]:


y_test = y_test.reshape(-1,1)
y_pred = y_pred.reshape(-1,1)
y_test.flatten()
y_pred.flatten()
#Now compare the actual output values for X_test with the predicted values, execute the following script:
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df


# In[ ]:





# In[ ]:





# In[ ]:




