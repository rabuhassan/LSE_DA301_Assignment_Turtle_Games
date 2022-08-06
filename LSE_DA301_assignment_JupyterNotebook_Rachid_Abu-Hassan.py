#!/usr/bin/env python
# coding: utf-8

# # Turtle Games - *Utilising Custmer Trends to Improve Sales*
# ## Scenario
# Turtle Games is a game manufacturer and retailer with a global customer base. The company manufactures and sells its own products, along with sourcing and selling products manufactured by other companies. Its product range includes books, board games, video games, and toys. The company collects data from sales as well as customer reviews. Turtle Games has a business objective of improving overall sales performance by utilising customer trends. 
# 
# To improve overall sales performance, Turtle Games has come up with an initial set of questions, which they would like help exploring.
# 
# 1. How customers accumulate loyalty points?
# 2. How groups within the customer base can be used to target specific market segments?
# 3. How social data (e.g. customer reviews) can be used to inform marketing campaigns?
# 4. The impact that each product has on sales?
# 5. How reliable the data is (e.g. normal distribution, skewness, or kurtosis)?
# 6. What the relationship(s) is/are (if any) between North American, European, and global sales?
#  

# ------

# ## 1. How Do Customers Accumulate Loyalty Points? 
# Investigate the possible relationships between the loyalty points, age, remuneration, and spending scores.

# ### 1.1 Workstation Setup

# In[2]:


# Import Necessary Packages.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
from statsmodels.formula.api import ols


# In[7]:


# Load the 'turtle_reviews' database and create the DataFrame. 
reviews = pd.read_csv('turtle_reviews.csv')


# ### 1.2 Explore the Data

# In[8]:


# View the DataFrame.
reviews.head()


# In[12]:


# Check for missing Values.
reviews.isnull().sum()


# In[11]:


# Explore the data.
reviews.info()


# In[10]:


# Descriptive statistics.
reviews.describe()


# ### 1.3 Clean-up the DataFrame and Export Clean csv

# In[18]:


# Drop unnecessary columns 'language' and 'platform'.
reviews.drop(['language', 'platform'], axis=1, inplace=True)


# In[22]:


# Change column headings:
# remunereation (k£) to salary, 
# spending_score (1-100) to spending_score).
reviews.rename(columns={'remuneration (k£)': 'salary', 
                        'spending_score (1-100)': 'spending_score'}, inplace=True)


# In[24]:


# Sense-check the DataFrame.
reviews.info()


# In[26]:


# Export the clean csv 'reviews' file. 
reviews.to_csv('reviews_clean.csv') 


# ### 1.4 Linear Regression
# Evaluate possible linear relationships between:
# * Loyalty Points and Age
# * Loyalty Points and Salary
# * Loyalty Points and Spending Score

# #### 1.41 Loyalty Points and Age

# In[29]:


# Define the variables: 
# Dependent Variable
y = reviews['loyalty_points']

# Independent Variable
x = reviews['age']

# Quick check for linearity.
plt.scatter(x, y)


# In[31]:


# OLS Model and Summary
# Create formula and pass through OLS methods.
f = 'y ~ x'
test = ols(f, data = reviews).fit()

# Print the regression table.
test.summary()


# In[32]:


# Extract the estimated parameters.
print("Parameters: ", test.params) 


# In[33]:


# Extract the standard errors.
print("Standard errors: ", test.bse)


# In[34]:


# Extract the predicted values.
print("Predicted values: ", test.predict())


# In[39]:


# Set the the x coefficient to ‘-4.012805’ 
# and the constant to ’1736.517739’ to generate the regression table.
y_pred = -4.012805 * reviews['age'] + 1736.517739

y_pred


# In[40]:


# Plot the graph with a regression line.

# Plot the data points.
plt.scatter(x,y)  

# Plot the regression line (in black).
plt.plot(x,y_pred, color='black') 

# Set the x and y limits on the axes:
plt.xlim(0)
plt.ylim(0)
plt.show()


# #### 1.42 Loyalty Points and Salary

# In[49]:


# Define the variables: 
# Dependent Variable
y = reviews['loyalty_points']

# Independent Variable
x = reviews['salary']

# Quick check for linearity.
plt.scatter(x, y)


# In[50]:


# OLS Model and Summary
# Create formula and pass through OLS methods.
f = 'y ~ x'
test = ols(f, data = reviews).fit()

# Print the regression table.
test.summary()


# In[51]:


# Extract the estimated parameters.
print("Parameters: ", test.params)


# In[52]:


# Extract the standard errors.
print("Standard errors: ", test.bse)


# In[53]:


# Extract the predicted values.
print("Predicted values: ", test.predict())


# In[54]:


# Set the the x coefficient to ‘34.187825’ 
# and the constant to ’-65.686513’ to generate the regression table.
y_pred = 34.187825 * reviews['salary'] - 65.686513

y_pred


# In[55]:


# Plot the graph with a regression line.

# Plot the data points.
plt.scatter(x,y)  

# Plot the regression line (in black).
plt.plot(x,y_pred, color='black') 

# Set the x and y limits on the axes:
plt.xlim(0)
plt.ylim(0)
plt.show()


# #### 1.43 Loyalty Points and Spending Score

# In[56]:


# Define the variables: 
# Dependent Variable
y = reviews['loyalty_points']

# Independent Variable
x = reviews['spending_score']

# Quick check for linearity.
plt.scatter(x, y)


# In[57]:


# OLS Model and Summary
# Create formula and pass through OLS methods.
f = 'y ~ x'
test = ols(f, data = reviews).fit()

# Print the regression table.
test.summary()


# In[58]:


# Extract the estimated parameters.
print("Parameters: ", test.params)


# In[59]:


# Extract the standard errors.
print("Standard errors: ", test.bse)


# In[60]:


# Extract the predicted values.
print("Predicted values: ", test.predict())


# In[61]:


# Set the the x coefficient to ‘33.061693’ 
# and the constant to ’-75.052663’ to generate the regression table.
y_pred = 33.061693 * reviews['spending_score'] - 75.052663

y_pred


# In[62]:


# Plot the graph with a regression line.

# Plot the data points.
plt.scatter(x,y)  

# Plot the regression line (in black).
plt.plot(x,y_pred, color='black') 

# Set the x and y limits on the axes:
plt.xlim(0)
plt.ylim(0)
plt.show()


# ### 1.5 Observations and Insights
# 
# Based on the data analysis and the linear regression models built to assess relationships between loyalty points and age/salary/spending score, the following was determined. 
# 
# 1. Age has little to no bearing on loyalty points and should not be used as a predictor. This can be clearly noted by a simple visual inspection of the scatterplot. Further, the coefficient of determination or r-sqrt 0.002, confirming no linear relationship exists between age and loyatly points. 
# 2. Both salary and spending score can be used to predict loyalty points. A linear relationship can be seen by visually inspecting both scatterplots. Further, the calcuated coefficient of determination for each is as follows: 
#   * Loyalty Points and Salary = r-sqrd is 0.380
#   * Loyalty Points and Spending Score = r-sqrd is 0.452

# ## 2. How groups within the customer base can be used to target specific market segments?
# Examine the usefulness of remuneration and spending scores in providing data for analysis. Identify groups within the customer base that can be used to target specific market segments. 

# ### 2.1 Workstation Setup (additional)

# In[110]:


# Import Additional Necessary Packages.
import matplotlib.cm as cm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings('ignore')


# In[125]:


# Load the 'reviews_clean' database and create the DataFrame df2. 
df2 = pd.read_csv('reviews_clean.csv', 
                  index_col=None)


# ### 2.2 Explore the Data and Reduce the DataFrame to Necessary Columns

# In[126]:


# View the DataFrame. 
df2.head()


# In[127]:


# Drop all columns except 'salary' and 'spending_score'.
df2.drop(['Unnamed: 0', 'gender', 'age', 'loyalty_points', 'education', 'product', 'review', 'summary'], 
             axis=1, 
             inplace=True)
df2.head()


# In[128]:


df2.info()


# In[129]:


# Descriptive statistics.
df2.describe()


# ### 2.3 Create Initial Plots

# In[131]:


# Create a scatterplot with Seaborn.
sns.scatterplot(x='salary', 
                y='spending_score', 
                data=df2)


# In[147]:


# Create a pairplot with Seaborn.
x = df2[['salary', 'spending_score']]

sns.pairplot(df2, 
             vars=x, 
             diag_kind= 'kde')


# ### 2.4 Determine the Number of Clusters

# #### 2.41 Elbow Method

# In[148]:


# Elbow chart to decide on the number of optimal clusters.
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    cs.append(kmeans.inertia_)

plt.plot(range(1, 11), cs, marker='o')
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("CS")

plt.show()


# #### 2.42 Silhoutte Method

# In[149]:


# Find the range of clusters to be used using silhouette method.
sil = []
kmax = 10

for k in range(2, kmax+1):
    kmeans_s = KMeans(n_clusters = k).fit(x)
    labels = kmeans_s.labels_
    sil.append(silhouette_score(x, labels, metric = 'euclidean'))

# Plot the silhouette method.
plt.plot(range(2, kmax+1), sil, marker='o')

plt.title("The Silhouette Method")
plt.xlabel("Number of clusters")
plt.ylabel("Sil")

plt.show()


# #### 2.43 Conclusion
# Evaluate clusters is **4, 5, and 6**; although, it is clear from the Elbow Method that 5 clusters is best. 

# ### 2.5 Evaluate k-means model at 4, 5, and 6

# #### 2.51 Evaluate and Fit 4 Clusters

# In[197]:


# Use 4 clusters:
kmeans = KMeans(n_clusters = 4, max_iter = 15000, init='k-means++', random_state=0).fit(x)
clusters = kmeans.labels_
x['K-Means Predicted'] = clusters

# Plot the predicted.
sns.pairplot(x, hue='K-Means Predicted', diag_kind= 'kde')


# In[198]:


# Check the number of observations per predicted class.
x['K-Means Predicted'].value_counts()


# In[199]:


# View the K-Means predicted.
print(x.head())


# In[200]:


# Visualising the clusters.
# Set plot size.
sns.set(rc = {'figure.figsize':(12, 8)})

sns.scatterplot(x='salary' , 
                y='spending_score',
                data=x , hue='K-Means Predicted',
                palette=['red', 'green', 'blue', 'orange'])


# #### 2.52 Evaluate and Fit 5 Clusters

# In[201]:


# Use 5 clusters:
kmeans = KMeans(n_clusters = 5, max_iter = 15000, init='k-means++', random_state=0).fit(x)
clusters = kmeans.labels_
x['K-Means Predicted'] = clusters

# Plot the predicted.
sns.pairplot(x, hue='K-Means Predicted', diag_kind= 'kde')


# In[202]:


# Check the number of observations per predicted class.
x['K-Means Predicted'].value_counts()


# In[203]:


# View the K-Means predicted.
print(x.head())


# In[204]:


# Visualising the clusters.
# Set plot size.
sns.set(rc = {'figure.figsize':(12, 8)})

sns.scatterplot(x='salary' , 
                y='spending_score',
                data=x , hue='K-Means Predicted',
                palette=['red', 'green', 'blue', 'orange', 'brown'])


# #### 2.53 Evaluate and Fit 6 Clusters

# In[205]:


# Use 6 clusters:
kmeans = KMeans(n_clusters = 6, max_iter = 15000, init='k-means++', random_state=0).fit(x)
clusters = kmeans.labels_
x['K-Means Predicted'] = clusters

# Plot the predicted.
sns.pairplot(x, hue='K-Means Predicted', diag_kind= 'kde')


# In[206]:


# Check the number of observations per predicted class.
x['K-Means Predicted'].value_counts()


# In[207]:


# View the K-Means predicted.
print(x.head())


# In[208]:


# Visualising the clusters.
# Set plot size.
sns.set(rc = {'figure.figsize':(12, 8)})

sns.scatterplot(x='salary' , 
                y='spending_score',
                data=x , hue='K-Means Predicted',
                palette=['red', 'green', 'blue', 'orange', 'brown', 'yellow'])


# ### 2.6 Fit the Final Model (k=5)

# In[209]:


# Use 5 clusters:
kmeans = KMeans(n_clusters = 5, max_iter = 15000, init='k-means++', random_state=0).fit(x)
clusters = kmeans.labels_
x['K-Means Predicted'] = clusters

# Plot the predicted.
sns.pairplot(x, hue='K-Means Predicted', diag_kind= 'kde')


# In[210]:


# Check the number of observations per predicted class.
x['K-Means Predicted'].value_counts()


# In[211]:


# View the K-Means predicted.
print(x.head())


# In[212]:


# Visualising the clusters.
# Set plot size.
sns.set(rc = {'figure.figsize':(12, 8)})

sns.scatterplot(x='salary' , 
                y='spending_score',
                data=x , hue='K-Means Predicted',
                palette=['red', 'green', 'blue', 'orange', 'brown'])


# ### 2.6 Observations and Insights
# 
# Based on the Elbow and Silhoutte Methods, it was clear that 5 clusters are best for the dataset considered. That was also visually confirmed early on by using a scatterplot of the salary and spending_score data. The plot was already showing more or less 5 clear clusters. 
# 
# Nonetheless, for good measure, 3 different cluster (k-values) were evaluated; 4, 5, and 6 clusters. Further, the all cluster values were evaluated, fitted, and ploted. This, further confirmed that 5 clusters is best for the dataset. 
# 
# The 5 clusters consist of the following numbers: 
# * 0    356
# * 1    774
# * 2    330
# * 3    269
# * 4    271
# 
# Clearly, cluster 1 is the largers with almost double any other cluster. Salaries within this cluster raged from 30k to 55k, while the spending score ranges from 40 to 60. 

# In[ ]:




