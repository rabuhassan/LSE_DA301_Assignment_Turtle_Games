# LSE_DA301_Assignment_Turtle_Games

# Turtle Games - *Utilising Custmer Trends to Improve Sales*
## Scenario
Turtle Games is a game manufacturer and retailer with a global customer base. The company manufactures and sells its own products, along with sourcing and selling products manufactured by other companies. Its product range includes books, board games, video games, and toys. The company collects data from sales as well as customer reviews. Turtle Games has a business objective of improving overall sales performance by utilising customer trends. 

To improve overall sales performance, Turtle Games has come up with an initial set of questions, which they would like help exploring.

1. How customers accumulate loyalty points?
2. How groups within the customer base can be used to target specific market segments?
3. How social data (e.g. customer reviews) can be used to inform marketing campaigns?
4. The impact that each product has on sales?
5. How reliable the data is (e.g. normal distribution, skewness, or kurtosis)?
6. What the relationship(s) is/are (if any) between North American, European, and global sales?
------------------------

## 1. How customers accumulate loyalty points?
Based on the data analysis and the linear regression models built to assess relationships between loyalty points and age/salary/spending score, the following was determined. 

1. Age has little to no bearing on loyalty points and should not be used as a predictor. This can be clearly noted by a simple visual inspection of the scatterplot. Further, the coefficient of determination or r-sqrt 0.002, confirming no linear relationship exists between age and loyatly points. 
2. Both salary and spending score can be used to predict loyalty points. A linear relationship can be seen by visually inspecting both scatterplots. Further, the calcuated coefficient of determination for each is as follows: 
  * Loyalty Points and Salary = r-sqrd is 0.380
  * Loyalty Points and Spending Score = r-sqrd is 0.452

## 2. How groups within the customer base can be used to target specific market segments?
Based on the Elbow and Silhoutte Methods, it was clear that 5 clusters are best for the dataset considered. That was also visually confirmed early on by using a scatterplot of the salary and spending_score data. The plot was already showing more or less 5 clear clusters. 

Nonetheless, for good measure, 3 different cluster (k-values) were evaluated; 4, 5, and 6 clusters. Further, the all cluster values were evaluated, fitted, and ploted. This, further confirmed that 5 clusters is best for the dataset. 

The 5 clusters consist of the following numbers: 
* 0    356
* 1    774
* 2    330
* 3    269
* 4    271

Clearly, cluster 1 is the largers with almost double any other cluster. Salaries within this cluster raged from 30k to 55k, while the spending score ranges from 40 to 60. 


## 3. Customer Sentiments with Reviews
Each of the surveyed customers provided a review and summary. Both sets of feedback were analyzed independently through the use of Natural Language Processing in order to assess customer statisfaction and extract insights that could be meaningful for the team at Turtle, and can inform their decision making process, customer retention, and help in addressing any issues with regards to their products and customer experience. 

* Initially, a two WordCloud plots were created for the Reviews and the Summaries in order to visually identify most used words and quickly assess the overall feedback. However, the amount of stopwords in use deemed this initial plots meaningless and further analysis needed to be done after the data had been further cleaned and filtered. 
* Once the data was cleaned by removing all stopwords and alphanumeric characters, two new WordCloud plots were generated. At first glance, the plots revealed that within the top words used were many words considered to be positive such as "fun", "great", etc. The analysis was taken a step further through the creation of count plots, to give a more acturate view of the most used words. 

Lastly, sentiment polarity was reviewed for both, the reviews and the summary feedback and histograms where created for both. 
* Reviews: overall, the sentiment is neutral to positive, with the most reviews falling just over 0. 
* Summary: overall, the most were neutral. However, the overall sentiment is positive leaning. 

Examination of the top 20 positive and negative revealed that some of the negative comments may be easily corrected through follow-ups and communication with the customers. 


