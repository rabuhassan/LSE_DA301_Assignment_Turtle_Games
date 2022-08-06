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
