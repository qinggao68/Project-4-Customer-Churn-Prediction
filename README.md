# Purpose
Since acquiring a new customer is 5x to 25x more expensive than retaining an existing one, I hope to use machine learning models to find the patterns within a given data set to see which factors contribute to customer churning. Then based on the patterns found in the dataset, we can identify which current customer is more at risk of churning and assist the operation team to develop a strategic road map for better managing both current and potential customers. 

# Key Findings
This analysis is studied upon a bank data set provided on the UCI datasets repository. Some of the key findings I found: 

- most common characteristics of customers that have already churned have: Fair FICO Score(6.9%), female(11.39%), highest income bracket($149k-$200k), older adults who are age 44-92(10.44%), been customer for less than 36 months(7.41%), Germany(8.14%), only have 1 product, not active member(13.02%), and has credit card(14.24%)
- based on the correlation matrix, we can see that predictors Age & Balance are mostly correlated with response variable - Exited, EstimatedSalary has almost no correlation, and CreditScore, Tenure, NumOfProducts, HasCrCard, and IsActiveMember moved in the opposite direction
![correlation matrix](https://github.com/qinggao68/Project-4-Customer-Churn-Prediction/blob/main/correlation_matrix.PNG)
- the top predictors that are most statistically signicant considered by all 6 models are: Germany, Male, Age, Balance, IsActiveMember, CreditScore, and NumOfProducts

![predictors selected by stepwise](https://github.com/qinggao68/Project-4-Customer-Churn-Prediction/blob/main/stepwise_significant_pred.PNG)
![predictors selected by rf](https://github.com/qinggao68/Project-4-Customer-Churn-Prediction/blob/main/important_predictors_selected_rf.PNG)

# Additional Information 
Some brief background of each machine learning model and definitions of the classification metrics that are used in this analysis: 

**Machine Learning Models:** 
1. Logistic Regression Model: used to model the probability of a certain class or event happening(e.g. probability of a customer churning)
2. Decision Tree Model: a model that partition data into subsets which each subset contains instances with similar values(only one tree is build)
3. Random Forest Model: a model composed of many decision trees and output is determined based on an average or a vote of all decision trees' results
4. Support Vector Machine: a model that use a hyperplane to separate data points into two classes and perform classification 

**Stepwise Logistic Regression:** it is similar to the logistic regression, except I used a variable selection method called StepWise to first pick out the top 6 most important predictors( in terms of what the model thinks the factor would affect whether a customer or churn or not). The stepwise method is an upgrade version of the forward and backward selection when it comes to variable selection. 

**Random-Over-Sampling-Examples(ROSE)**: is a method to re-sample data as in the data we have more current customers than customers that churned. Since there is less minority class(customers that already churned), without resampling, the accuracy will be lower. So, ROSE is an upgrade resampling methods than the undersampling and oversampling methods in which it create a sample of synthetic data. 

# Project Scope 

1. Examine the data set provided and check to see if there are any missing values 
2. Create visualization for both categorical and numerical variables and compare the relationship to the response variable - customer status
3. Correlation matrix
4. Scale data
5. Split data into train and validate 
6. Access if data are imbalance or not
7. Build a Logistic Regression using all predictors 
8. Build a Stepwise Logistic Regression to select the most significant predictors 
9. Resample data using Random Over-Sampling Examples to build a second Logistic Regression with only the most significant predictors 
10. Build a Decision Tree model and pruned the full decision tree
11. Build a Random Forest model 
12. Build a Support Vector Machine
13. Select the final model based on highest accuracy rate 

# Future Work 
I am currently working on the app, which will be uploaded to the Shiny server, and it will contain data visualization and prediction of a customer's churn rate. 

I also plan to hyperparameter tuning the models I built to improve the accuracy rate. 

# Reference
[TrustTheData's Kaggle Notebook done using Python] https://www.kaggle.com/code/kmalit/bank-customer-churn-prediction
[Atindra Bandi's Kaggle Notebook done using R on a different data set] 
https://www.kaggle.com/code/janiobachmann/anticipating-attrition-automl-to-the-rescue/report 
[stackoverflow's code on drawing correlation matrix]  https://stackoverflow.com/questions/37897252/plot-confusion-matrix-in-r-using-ggplot
