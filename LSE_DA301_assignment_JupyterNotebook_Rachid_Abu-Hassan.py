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

# In[8]:


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

# ---

# ## 3. Analysis of customer sentiments with reviews.
# Identify: 
# * the 15 most common words used in online product reviews
# * the top 20 positive reviews and the top 20 negative reviews received from the website.

# ### 3.1 Workstation Setup (additional)

# In[2]:


# Import Additional Necessary Packages.
import nltk 
import os

nltk.download ('punkt')
nltk.download ('stopwords')

from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from textblob import TextBlob
from scipy.stats import norm

# Import Counter.
from collections import Counter

import warnings
warnings.filterwarnings('ignore')


# In[18]:


# Load the 'reviews_clean' database and create the DataFrame df3. 
df3 = pd.read_csv('reviews_clean.csv', 
                  index_col=None)


# ### 3.2 Explore the Data and Reduce the DataFrame to Necessary Columns

# In[19]:


# View DataFrame. 
df3.head()


# In[20]:


df3.info()


# In[21]:


# Drop all columns except 'salary' and 'spending_score'.
df3.drop(['Unnamed: 0', 'gender', 'age', 'salary', 'spending_score', 'loyalty_points', 'education', 'product'], 
             axis=1, 
             inplace=True)
df3.head()


# In[24]:


# Check for missing values. 
df3.isnull().sum()


# ### 3.3 Data Preperation for NLP

# #### 3.31 Change to Lowercase and Join Elements

# * **review Column**

# In[28]:


# Change all words in the Review Column to lowercase and join with a space. 
df3['review'] = df3['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Preview the result.
df3['review'].head()


# * **summary Column**

# In[29]:


# Change all words in the Summary Column to lowercase and join with a space. 
df3['summary'] = df3['summary'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Preview the result.
df3['summary'].head()


# #### 3.32 Replace Punctuation

# * **review Column**

# In[31]:


# Remove punctuation in Review Column.
df3['review'] = df3['review'].str.replace('[^\w\s]','')

# Preview the result.
df3['review'].head()


# * **summary Column**

# In[32]:


# Remove punctuation in Review Column.
df3['summary'] = df3['summary'].str.replace('[^\w\s]','')

# Preview the result.
df3['summary'].head()


# #### 3.33 Remove Duplicates in Both Columns

# * **review Column**

# In[34]:


# Check the number of duplicate.
df3.review.duplicated().sum()


# In[35]:


# Drop the duplicates.
df3 = df3.drop_duplicates(subset=['review'])


# * **summary Column**

# In[42]:


# Check the number of duplicate.
df3.summary.duplicated().sum()


# In[43]:


# Drop the duplicates.
df3 = df3.drop_duplicates(subset=['summary'])


# * **Reset Index**

# In[44]:


# Preview data.
df3.reset_index(inplace=True)
df3.head()


# In[51]:


# Drop 'level_0 column'.
df3.drop(['level_0'], 
             axis=1, 
             inplace=True)
df3.head()


# ### 3.4 Tokenise and Create Wordclouds

# In[55]:


# Create a copy of the DataFrame. 
df4 = df3.copy()

# View the DataFrame. 
df4.head()


# #### 3.41 Tokenise Both Columns

# * **review Column**

# In[61]:


# Tokenise the words in review. 
df4['review_tokens'] = df4['review'].apply(word_tokenize)

# Preview data.
df4['review_tokens'].head()


# * **summary Column**

# In[62]:


# Tokenise the words in review. 
df4['summary_tokens'] = df4['summary'].apply(word_tokenize)

# Preview data.
df4['summary_tokens'].head()


# #### 3.42 Create Wordclouds

# * **Review**

# In[73]:


# String all the reviews together in a single variable.
# Create an empty string variable.
all_reviews = ''
for i in range(df4.shape[0]):
    # Add each review comment.
    all_reviews = all_reviews + df4['review'][i]


# In[74]:


# Set the colour palette.
sns.set(color_codes=True)

# Create a WordCloud object.
review_word_cloud = WordCloud(width = 1600, height = 900, 
                              background_color ='white',
                              colormap = 'plasma', 
                              stopwords = 'none',
                              min_font_size = 10).generate(all_reviews)


# In[75]:


# Plot the Reivew WordCloud image.                    
plt.figure(figsize = (16, 9), facecolor = None) 
plt.imshow(review_word_cloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.savefig('review_w_stopwords.png')
plt.show()


# * **Summary**

# In[76]:


# String all the summary comments together in a single variable.
# Create an empty string variable.
all_summaries = ''
for i in range(df4.shape[0]):
    # Add each summary comment.
    all_summaries = all_summaries + df4['summary'][i]


# In[77]:


# Set the colour palette.
sns.set(color_codes=True)

# Create a WordCloud object.
summary_word_cloud = WordCloud(width = 1600, height = 900, 
                               background_color ='white',
                               colormap = 'plasma', 
                               stopwords = 'none',
                               min_font_size = 10).generate(all_summaries)


# In[78]:


# Plot the Summary WordCloud image.                    
plt.figure(figsize = (16, 9), facecolor = None) 
plt.imshow(review_word_cloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.savefig('summary_w_stopwords.png')
plt.show()


# ### 3.5 Frequency Distribution and Polarity

# #### 3.51 Frequency Distribution

# * **Review**

# In[82]:


# Define an empty list of review tokens.
all_review_tokens = []

for i in range(df4.shape[0]):
    # Add each token to the list.
    all_review_tokens = all_review_tokens + df4['review_tokens'][i]


# In[83]:


# Calculate the frequency distribution.
review_fdist = FreqDist(all_review_tokens)

# Preview data.
review_fdist


# * **Summary**

# In[84]:


# Define an empty list of summary tokens.
all_summary_tokens = []

for i in range(df4.shape[0]):
    # Add each token to the list.
    all_summary_tokens = all_summary_tokens + df4['summary_tokens'][i]


# In[85]:


# Calculate the frequency distribution.
summary_fdist = FreqDist(all_summary_tokens)

# Preview data.
summary_fdist


# #### 3.52 Remove Alphanumeric Characters and Stopwords

# In[87]:


# Delete all the alpanum.
all_summary_tokens_1 = [word for word in all_summary_tokens if word.isalnum()]
all_review_tokens_1 = [word for word in all_review_tokens if word.isalnum()]


# In[88]:


# Create a set of English stop words.
english_stopwords = set(stopwords.words('english'))

# Create a filtered list of tokens without stop words.
summary_tokens = [x for x in all_summary_tokens_1 if x.lower() not in english_stopwords]
review_tokens = [x for x in all_review_tokens_1 if x.lower() not in english_stopwords]


# In[89]:


# Define an empty string variable.
summary_tokens_string = ''

for value in summary_tokens:
    # Add each filtered token word to the string.
    summary_tokens_string = summary_tokens_string + value + ' '


# In[90]:


# Define an empty string variable.
review_tokens_string = ''

for value in review_tokens:
    # Add each filtered token word to the string.
    review_tokens_string = review_tokens_string + value + ' '


# #### 3.53 Create Worldcloud without Stopwords

# * **Reivew**

# In[91]:


# Set the colour palette.
sns.set(color_codes=True)

# Create a WordCloud object.
review_final_wordcloud = WordCloud(width = 1600, height = 900, 
                                   background_color ='white',
                                   colormap = 'plasma', 
                                   stopwords = 'none',
                                   min_font_size = 10).generate(review_tokens_string)


# In[92]:


# Plot the Reivew WordCloud image.                    
plt.figure(figsize = (16, 9), facecolor = None) 
plt.imshow(review_final_wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.savefig('review_wo_stopwords.png')
plt.show()


# * **Summary**

# In[94]:


# Set the colour palette.
sns.set(color_codes=True)

# Create a WordCloud object.
summary_final_wordcloud = WordCloud(width = 1600, height = 900, 
                                    background_color ='white',
                                    colormap = 'plasma', 
                                    stopwords = 'none',
                                    min_font_size = 10).generate(summary_tokens_string)


# In[95]:


# Plot the Summary WordCloud image.                    
plt.figure(figsize = (16, 9), facecolor = None) 
plt.imshow(summary_final_wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.savefig('summary_wo_stopwords.png')
plt.show()


# #### 3.54 What are the 15 Most Common Words and Polarity

# * **Review**

# In[97]:


# View the frequency distribution.
review_dist = FreqDist(review_tokens)

# Preview the data.
review_dist


# In[99]:


# Generate a DataFrame from Counter for Review Words.
review_counts = pd.DataFrame(Counter(review_tokens).most_common(15), 
                             columns=['Word', 'Frequency']).set_index('Word')

# Preview data.
review_counts


# In[104]:


# Set the plot type.
ax = review_counts.plot(kind='barh', figsize=(16, 9), fontsize=12, 
                        colormap ='plasma')

# Set the labels.
ax.set_xlabel('Count', fontsize=12)
ax.set_ylabel('Word', fontsize=12)
ax.set_title("REVIEW: Count of the 15 most frequent words",
             fontsize=20)

# Draw the bar labels.
for i in ax.patches:
    ax.text(i.get_width()+.41, i.get_y()+.1, str(round((i.get_width()), 2)),
            fontsize=12, color='red')

# Save plot.
plt.savefig('review_word_count.png')


# * **Summary**

# In[98]:


# View the frequency distribution.
summary_dist = FreqDist(summary_tokens)

# Preview the data.
summary_dist


# In[100]:


# Generate a DataFrame from Counter for Review Words.
summary_counts = pd.DataFrame(Counter(summary_tokens).most_common(15), 
                              columns=['Word', 'Frequency']).set_index('Word')

# Preview data.
summary_counts


# In[105]:


# Set the plot type.
ax = summary_counts.plot(kind='barh', figsize=(16, 9), fontsize=12,
                         colormap ='plasma')

# Set the labels.
ax.set_xlabel('Count', fontsize=12)
ax.set_ylabel('Word', fontsize=12)
ax.set_title("SUMMARY: Count of the 15 most frequent words",
             fontsize=20)

# Draw the bar labels.
for i in ax.patches:
    ax.text(i.get_width()+.41, i.get_y()+.1, str(round((i.get_width()), 2)),
            fontsize=12, color='red')
    

# Save plot.
plt.savefig('summary_word_count.png')


# ### 3.6 Polarity and Sentiment

# In[106]:


# Define a function to extract a polarity score for the comment.
def generate_polarity(comment):
    return TextBlob(comment).sentiment[0]


# In[107]:


# Populate a new column with polarity scores for each comment.
df4['review_polarity'] = df4['review'].apply(generate_polarity)
df4['summary_polarity'] = df4['summary'].apply(generate_polarity)

# View DataFrame.
df4.head()


# #### 3.61 REVIEW - Sentiment Polarity Scores Histogram

# In[109]:


# Set the number of bins.
num_bins = 15

# Set the plot area.
plt.figure(figsize=(16,9))

# Define the bars.
n, bins, patches = plt.hist(df4['review_polarity'], num_bins, facecolor='red', alpha=0.6)

# Set the labels.
plt.xlabel('Polarity', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('REVIEW: Histogram of sentiment score polarity', fontsize=20)

# Save plot.
plt.savefig('Review_Hist_Polarity.png')
plt.show()


# #### 3.62 SUMMARY - Sentiment Polarity Scores Histogram

# In[110]:


# Set the number of bins.
num_bins = 15

# Set the plot area.
plt.figure(figsize=(16,9))

# Define the bars.
n, bins, patches = plt.hist(df4['summary_polarity'], num_bins, facecolor='red', alpha=0.6)

# Set the labels.
plt.xlabel('Polarity', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('SUMMARY: Histogram of sentiment score polarity', fontsize=20)

# Save plot.
plt.savefig('Summary_Hist_Polarity.png')
plt.show()


# ### 3.6 Top 20 Positive and Negative Reviews and Summaries

# #### 3.61 Negatives

# * **Reviews**

# In[111]:


# Create a DataFrame.
negative_sentiment_reviews = df4.nsmallest(20, 'review_polarity')

# Eliminate unnecessary columns.
negative_sentiment_reviews = negative_sentiment_reviews[['review', 'review_polarity']]

# Eliminate unnecessary columns.
negative_sentiment_reviews.style.set_properties(subset=['review'], **{'width': '1200px'})


# * **Summary**

# In[113]:


# Create a DataFrame.
negative_sentiment_summary = df4.nsmallest(20, 'summary_polarity')

# Eliminate unnecessary columns.
negative_sentiment_summary = negative_sentiment_summary[['summary', 'summary_polarity']]

# Eliminate unnecessary columns.
negative_sentiment_summary.style.set_properties(subset=['summary'], **{'width': '1200px'})


# #### 3.62 Positives

# * **Reviews**

# In[115]:


# Create a DataFrame.
positive_sentiment_reviews = df4.nlargest(20, 'review_polarity')

# Eliminate unnecessary columns.
positive_sentiment_reviews = positive_sentiment_reviews[['review', 'review_polarity']]

# Eliminate unnecessary columns.
positive_sentiment_reviews.style.set_properties(subset=['review'], **{'width': '1200px'})


# * **Summary**

# In[117]:


# Create a DataFrame.
positive_sentiment_summary = df4.nlargest(20, 'summary_polarity')

# Eliminate unnecessary columns.
positive_sentiment_summary = positive_sentiment_summary[['summary', 'summary_polarity']]

# Eliminate unnecessary columns.
positive_sentiment_summary.style.set_properties(subset=['summary'], **{'width': '1200px'})


# ### 3.7 Observations and Insights
# 
# Each of the surveyed customers provided a review and summary. Both sets of feedback were analyzed independently through the use of Natural Language Processing in order to assess customer statisfaction and extract insights that could be meaningful for the team at Turtle, and can inform their decision making process, customer retention, and help in addressing any issues with regards to their products and customer experience. 
# * Initially, a two WordCloud plots were created for the Reviews and the Summaries in order to visually identify most used words and quickly assess the overall feedback. However, the amount of stopwords in use deemed this initial plots meaningless and further analysis needed to be done after the data had been further cleaned and filtered. 
# * Once the data was cleaned by removing all stopwords and alphanumeric characters, two new WordCloud plots were generated. At first glance, the plots revealed that within the top words used were many words considered to be positive such as "fun", "great", etc. The analysis was taken a step further through the creation of count plots, to give a more acturate view of the most used words. 
# Lastly, sentiment polarity was reviewed for both, the reviews and the summary feedback and histograms where created for both. 
# * Reviews: overall, the sentiment is neutral to positive, with the most reviews falling just over 0. 
# * Summary: overall, the most were neutral. However, the overall sentiment is positive leaning. 
# Examination of the top 20 positive and negative revealed that some of the negative comments may be easily corrected through follow-ups and communication with the customers. 

# ---
