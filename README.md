# Udacity-Analyze_AB_Testing-
Udacity Data Analyst Nanodegree Project 3 : Analyze_AB_Testing

Project Overview :

For this project, I worked to understand the results of an A/B test run by an e-commerce
website. The company has developed a new web page in order to try and increase the number
of users. My goal was to help the company understand if they should implement this new
page, keep the old page, or perhaps run the experiment longer to make their decision.

To complete this project, i'll require the following softwares:
  - Python (Numpy, Pandas, Matplotlib, StatsModels)
  - Jupyter Notebook


Introduction
A/B tests are very commonly performed by data analysts and data scientists. Within the
framework this project, we tried to understand whether the company should implement a new
page or keep the old page with following:
  - Probability approach
  - A/B test
  - Regression approach

Part 1: Probability approach
  - It appears that individuals in the treatment group had a conversion rate of 11.88% and
  individuals in the control group had a conversion rate of 12.04%.
  - We found that probability of an individual receiving the new page is 0.50006
  - Meaning, there is almost the same chance that an individual received the old page

Part 2: A/B test
  - In A/B test we set up our hypothesis to test if new page results in better conversion or
  not
  - We simulated our user groups with respect to conversions
  - We found the p_value to be 0.4122
  - With such a p-value, we failed to reject null hypothesis
  - By using the built-in stats.proportions_ztest we computed z-score and p-value which
  confirmed our earlier p-value and failure to reject null hypothesis

Part 3: Regression Approach
  - We looked at exploring two possible outcomes. Whether new page is better or not.
  - With logistic regression results, we again encountered same z-score as well as p-value
  of 0.19, corresponding two-tailed case
  - By further adding geographic location of the users, we tried to find if any specific
  country had an impact on conversion
  - The result gave a similar outlook and suggested that the countries have no impact on
  the conversion rate.
