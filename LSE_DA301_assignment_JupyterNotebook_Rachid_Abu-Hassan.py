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

# In[ ]:




