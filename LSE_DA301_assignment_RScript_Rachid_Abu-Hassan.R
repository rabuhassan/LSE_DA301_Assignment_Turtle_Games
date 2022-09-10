## LSE Data Analytics Online Career Accelerator 

# DA301:  Advanced Analytics for Organisational Impact
# Rachid Abu-Hassan

###############################################################################

### Scenario ###

# Turtle Games is a game manufacturer and retailer with a global customer base. 
# The company manufactures and sells its own products, along with sourcing and 
# selling products manufactured by other companies. Its product range includes 
# books, board games, video games, and toys. The company collects data from 
# sales as well as customer reviews. Turtle Games has a business objective of 
# improving overall sales performance by utilising customer trends. 

# To improve overall sales performance, Turtle Games has come up with an 
# initial set of questions, which they would like help exploring.

# 1. How customers accumulate loyalty points? (Analysis completed in Python)
# 2. How groups within the customer base can be used to target specific market 
# segments? (Analysis complete in Python)
# 3. How social data (e.g. customer reviews) can be used to inform marketing 
# campaigns? (Analysis completed in Python)
# 4. The impact that each product has on sales?
# 5. How reliable the data is (e.g. normal distribution, skewness, or 
# kurtosis)?
# 6. What the relationship(s) is/are (if any) between North American, European, 
# and global sales?

################################################################################

#
##
###
####
##### 4. The Impact that Each Product has on Sales ##### 
####
###
##
#

###############################################################################

### 4.1 Load and Explore the Data ###

# Install and import Tidyverse.
library('tidyverse')
# Create insightful summaries of the data set.
library(skimr)

# Import the data set 'turtle_sales.csv'.
sales <- read.csv(file.choose(), header=T)

# Print the data frame.
sales

# View the data frame. 
View(sales)

# Create a new data frame from a subset of the sales data frame.
# Remove unnecessary columns. 
sales2 <- select(sales, -Ranking, -Year, -Genre, -Publisher)

# View the data frame.
head(sales2)

# View the descriptive statistics.
summary(sales2)

# Description statistics show that the Product data column is treated
# as integer, while it should be factor, since categorical.
# Confirm ALL data types by converting data frame to a tibble. 
as_tibble(sales2)

# Check frequency of product to confirm more one row per product before 
# converting.
table(sales2$Product)

# Convert 'product' to factor (categorical variable).
sales3 <- mutate(sales2,
                 Product=as.factor(Product))

# Checked and confirmed in console that sales3 data frame is ready. 
# View the descriptive statistics for the 'sales3' data frame. 
summary(sales3)
# View the data frame once more. 
glimpse(sales3)


################################################################################

### 4.2 Create and Review Plots for Insights ###

View(sales3)

## 4.2.1 Scatterplots ##

# NA_Sales vs. EU_Sales to check if the sames platforms are popular in both 
# markets. 
qplot(NA_Sales, EU_Sales, 
      color = Platform, 
      data = sales3)

# Review both regions against Global Sales.
# Global_Sales vs. EU_Sales
qplot(Global_Sales, EU_Sales,
      color = Platform,
      data = sales3)
# Global_Sales vs. NA_Sales
qplot(Global_Sales, NA_Sales,
      color = Platform,
      data = sales3)

# Visualise sales per region and global to confirm best selling platform. 
# Global_Sales. 
qplot(Platform,
      Global_Sales,
      data=sales3)
# EU_Sales. 
qplot(Platform,
      EU_Sales,
      data=sales3)
# NA_Sales. 
qplot(Platform,
      NA_Sales,
      data=sales3)


## 4.2.2 Histograms ##
# Histograms for sales per region.
# EU_Sales. 
qplot(EU_Sales,
      data=sales3,
      geom = 'histogram',
      binwidth = 1)

# NA_Sales. 
qplot(NA_Sales,
      data=sales3,
      geom = 'histogram',
      binwidth = 1)

## 4.2.3 Create and Review Plots for Insights ##

# Create boxplots to view the distribution, skewness, and quartiles per 
# platform review the descriptive statistics of each of the sales variables; 
# Global_Sales, EU_Sales, and NA_Sales. 
# Global_Sales.
qplot(Global_Sales, Platform, 
      data = sales3,
      geom='boxplot')
# EU_Sales.
qplot(EU_Sales, Platform, 
      data = sales3,
      geom='boxplot')
# Global_Sales.
qplot(NA_Sales, Platform, 
      data = sales3,
      geom='boxplot')


###############################################################################

### 4.3 The Impact on Sales per Product ###

## 4.3.1 Group by Product and Determine the Sales Per Product ##
# Create a new data frame grouped by product.
sales4 <- aggregate(cbind(NA_Sales, EU_Sales, Global_Sales) ~ Product, 
                    sales3, 
                    FUN = sum)

# View data frame. 
View(sales4)

# Explore 
summary(sales4)

## 4.3.2 Create and Review Charts ##

# NA_Sales vs. EU_Sales to examine if the same games are popular in both
# regions.
# Create a scatter plot. 
qplot(EU_Sales,
      NA_Sales,
      data = sales4,
      geom = c('point', 'jitter'))

# Create histogram to highlight the most sales range. 
qplot(Global_Sales,
      data = sales4,
      binwidth = 3,
      geom = 'histogram')

# Create boxplots to view the distribution of the EU_Sales and NA_Sales. 
boxplot(sales4$NA_Sales)
boxplot(sales4$EU_Sales)

###############################################################################

### 4.4 Observations and Insights ###

## Based on the initial analysis conducted on the sales data, it appears that 
# there exists correlation between EU_Sales and NA_Sales; meaning, the same 
# products are equally as popular in both regions. 

# Attention should especially be paid to product 107 and 254, which if data is
# correct, are generating the most sales value. However, they are clearly 
# outliers based on the boxplots and the descriptive statistics. Therefore, 
# it is highly recommended that a discussion with Turtle is around the data 
# provided pertaining to these two products to ensure the data is sound.

# All in all, North America sales are higher than Europeann Sales.

# As for the platforms, products running on the Wii, the DS, the GB, and the 
# NES are seemingly bringing in higher sales numbers. 

# Most sale values in the EU fall under Pounds 2.5M. While in North American, 
# most sale values fall under Pounds 4.0M. 

###############################################################################

#
##
###
####
##### 5. Data Clean-up and Manipulation ##### 
####
###
##
#

################################################################################

### 5.1 Load and Explore the Data ###

# Re-View the data frame 'sales4', which includes the Product, Global_Sales, 
#EU_Sales, and NA_Sales. 
View(sales4)
head(sales4)

# Create a subset of the data frame with only numeric columns.
sales_values <- sales4 %>% keep(is.numeric)
sales_values

# Determine the min, max, and mean values.
# min
sales4_min <- sapply(sales_values, min)
sales4_min
# max
sales4_max <- sapply(sales_values, max)
sales4_max
# mean
sales4_mean <- sapply(sales_values, mean)
sales4_mean
# sum
sales4_sum <- sapply(sales_values, sum)
sales4_sum

# View the descriptive statistics.
summary(sales4)

###############################################################################

### 5.2 Determine the Normality ###

## 5.2.1 Create Q-Q Plots ##
# Create Q-Q Plots for Global_Sales, EU_Sales, and NA_Sales. 

# Q-Q plot Global_Sales
qqnorm(sales4$Global_Sales)
# Add a red reference line.
qqline(sales4$Global_Sales, col = 'red')

# Q-Q plot for EU_Sales
qqnorm(sales4$EU_Sales)
# Add a red reference line.
qqline(sales4$EU_Sales, col = 'red')

# Q-Q plot for NA_Sales
qqnorm(sales4$NA_Sales)
# Add a red reference line.
qqline(sales4$NA_Sales, col = 'red')


## 5.2.2 Perform Shapiro-Wilk Test ##
# Install and import Moments.
library(moments)

# Perform Shapiro-Wilk test on Global_Sales, EU_Sales, and NA_Sales.

# Shapiro-Wilk test on Global_Sales
shapiro.test(sales4$Global_Sales)
# p-value is well below 0.05. Not a normal distribution. 

# Shapiro-Wilk test on EU_Sales
shapiro.test(sales4$NA_Sales)
# p-value is well below 0.05. Not a normal distribution. 

# Shapiro-Wilk test on NA_Sales
shapiro.test(sales4$NA_Sales)
# p-value is well below 0.05. Not a normal distribution. 


## 5.2.3 Determine Skewness and Kurtosis ##
# Perform Skewness and Kurtosis test on Global_Sales, EU_Sales, and NA_Sales.

# Skewness and Kurtosis test on Global_Sales
skewness(sales4$Global_Sales)
kurtosis(sales4$Global_Sales)

# Positive skewness and a very high kurtosis value of 17 suggesting data is NOT
# platykurtic. 

# Skewness and Kurtosis test on EU_Sales
skewness(sales4$EU_Sales)
kurtosis(sales4$EU_Sales)

# Positive skewness and a very high kurtosis value of 16 suggesting data is NOT
# platykurtic. 

# Skewness and Kurtosis test on NA_Sales
skewness(sales4$NA_Sales)
kurtosis(sales4$NA_Sales)

# Positive skewness and a very high kurtosis value of 15 suggesting data is NOT
# platykurtic. 


## 5.2.4 Determine Correlation ##
# Determine correlation.
cor(sales4$EU_Sales, sales4$NA_Sales)

# A correlation coefficient of 0.62 suggests a moderate positive correlation.
# Meaning many top selling games in EU are also top selling in NA. 

###############################################################################

### 5.3 Plot the Data ###
# Create plots to gain insights into data.

ggplot(data=sales4,
       mapping=aes(x = EU_Sales, y = NA_Sales)) +
  geom_point(alpha=.5, size=3) +
  geom_smooth(method='lm', se=FALSE, size=1) + 
  scale_x_continuous(breaks=seq(0, 50, 5)) +
  scale_y_continuous(breaks=seq(0, 50, 5)) +
  labs(title="Relationship between European Sales and North American Sales",
       subtitle="Data obtained from Turtle's Website Sales",
       #  [3] Add labels to labs function.
       caption="Source: Turtle's Websales",
       x="European Sales (in Millions of Pounds)",
       y="North American Sales (in Millions of Pounds)")

###############################################################################

### 5.4 Observations and Insights ###

# Based on the various tests and analysis conducted, including but not limited,
# to Shipiro Test, Skewness and Kurosis, the following observations were made:

# The minimum global sales sum for a single product is 4.20. While the maximum 
# global sales ssum for a single product is 67.85. 

# The mean of sales sum in NA_Sales is Pounds 5.06M, in the EU_Sales is 
# pounds 3.31M, and Globally is pounds 10.730343M.  

# The Shipiro Testing for all sales produced a p-value is well below 0.05,
# suggesting the data is not normally distributed.

# Where as the all sales variables are postively skewed as expected, and show
# a very high kurtosis value of over 15 suggesting data. This is likely to 
# outliers where not removed on the premise that no discussion was conducted 
# with the client to confirm that these values are in correct and should 
# indeed be removed. 

# Lastly, at this stage of the analysis, a correlation coefficient of 0.62 was 
# calculated suggesting a moderate positive correlation and confirming what
# was witnessed during earlier analysis. Meaning many top 
# selling games in EU are also top selling in NA. 

###############################################################################

#
##
###
####
##### 6. Making Recommendations to the Business ##### 
####
###
##
#

###############################################################################

### 6.1 Load and Explor the Data ###

# View the data frame.
View(sales4)

# Run a summary of the data frame.
summary(sales4)

###############################################################################

### 6.2 Create a Simple Linear Regression Model ###

## 6.2.1 Determine the Correlation between Columns ##
# Create a linear regression model on the original data.
model1 <- lm(EU_Sales ~ NA_Sales, sales4)
summary(model1)

model2 <- lm(Global_Sales ~ EU_Sales, sales4)
summary(model2)

model3 <- lm(Global_Sales ~ NA_Sales, sales4)
summary(model3)


## 6.2.2 Plot (simple linear regression) ##
# Basic visualisation.

# model1
# Plot the residuals and add best-fit. 
plot(model1$residuals)
# Add line-of-best-fit.
abline(coefficients(model1))

# Calculate the sum of squares error (SSE) to determine strength.
SSE1 = sum(model1$residuals^2)

# View the result.
SSE1

# Very high SSE = model1 is a poor fit.
# The closer the SSE is to 0, the better the fit.


# model2
# Plot the residuals and add best-fit. 
plot(model2$residuals)
# Add line-of-best-fit.
abline(coefficients(model1))

# Calculate the sum of squares error (SSE) to determine strength.
SSE2 = sum(model2$residuals^2)

# View the result.
SSE2

# Very high SSE = model1 is a poor fit.
# The closer the SSE is to 0, the better the fit.


# model3
# Plot the residuals and add best-fit. 
plot(model3$residuals)
# Add line-of-best-fit.
abline(coefficients(model3))

# Calculate the sum of squares error (SSE) to determine strength.
SSE3 = sum(model3$residuals^2)

# View the result.
SSE3

# Very high SSE = model1 is a poor fit.
# The closer the SSE is to 0, the better the fit.

###############################################################################

### 6.3 Create a Multiple Linear Regression Model ###

# Multiple linear regression model.
model4 = lm(Global_Sales ~ EU_Sales + NA_Sales,
            data = sales4) 

summary(model4)

###############################################################################

### 6.4 Predictions Based on Given Values ###

# Create prepare test data.

sales_test <- subset(sales4, EU_Sales %in% c(23.80, 0.65, 0.97, 0.52))
print (sales_test)

# Compare with observed values for a number of records.
# Create a new object and specify the predict function.
predictTest = predict(model4, newdata = sales_test,
                      interval='confidence')

# Print the object.
predictTest 

###############################################################################

### 6.5 Observations and Insights ###

# Four different models were build; however, the best fit and most useful is 
# model4, which predicts Global Sales based on EU and North American Sales. 
# The adjusted r-squared for model4 is over 0.96; which can be expected, given 
# that Global Sales is the sum of North American Sales and European Sales. 
# Based on the model4, four different values were used as a test set to predict
# global sales. The following results where prediected and compared to the 
# actual values presented in the data base. 

# The first predicted value is 68.056548; the actual is 67.85. 
# The second predicted value is 4.908353; the actual is 4.32.
# The third predicted value is 7.202698; the actual is 6.12.
# The fourth predicted value is 26.625558; the actual is 23.21.

## Recommendations ##
# Firstly, it is highly recommended that disccusions take place with Turtle
# before these results are confirmed. That is mainly due to the confirming 
# the quality of the data and the presense of outliers. 

# Secondly, with sales in North America almost double those in Europe, 
# additional efforts should be put in place to increase sales in Europe. 

# The most grossing products, if indeed the data is accurate, suggests that 
# Sports products are high grossing; far more than any other products Turle is
# selling. Thus, Turtle may want to consider increasing sports product 
# advertising. 

# Lastly, it may also be good for Turtle to drop some products; namely those
# on the 2600, the GEN, the PSV, and the WiiU platforms. 


###############################################################################
###############################################################################




