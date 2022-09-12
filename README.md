# Logistic-Regression

Logistic regression estimates the probability of an event occurring, such as voted or didn’t vote, based on a given dataset of independent variables. 
Since the outcome is a probability, the dependent variable is bound between 0 and 1.

## Data Source ##

     The Salary Data dataset from Kaggle has been used and the model has been used and trained on attributes like 'Years of Experience' and 'Salary'.
         
## Attribute Table ##

S.No | Attribute | Remarks 
-----|-----------|--------
1 | Years of Experience | Experience of the Employee in number of years
2 | Salary | Wage of the Employee on the basis of years of experience

## Step 1: Importing python libraries ##
      
     import pandas as pd
     from matplotlib import pyplot as plt
     import matplotlib.pyplot as plt
     
## Step 2: Loading the Dataset ##

     df = pd.read_csv("Salary_Data.csv")
     df.head()
     
## Step 3: Plotting a scatterplot of the dataset ##
     
     plt.scatter(df.YearsExperience,df.Salary,marker='+',color='red')

## Step 4: Importing the train-test-split model ##

     from sklearn.model_selection import train_test_split
     
## Step 5: Training the model ##
   
     X_train, X_test, y_train, y_test = train_test_split(df[['YearsExperience']],df.Salary,train_size=0.8)
     X_test
     
## Step 6: Import the logistic Regression model ##
     
     from sklearn.linear_model import LogisticRegression
     
## Step 7: Fitting the model ##
     
     model = LogisticRegression()
     model.fit(X_train, y_train)
     X_test
     
## Step 8: Performing Predictions ##
     
     y_predicted = model.predict(X_test)
     model.predict_proba(X_test)
     model.score(X_test,y_test)
     y_predicted
     X_test
     model.coef_
     model.intercept_
