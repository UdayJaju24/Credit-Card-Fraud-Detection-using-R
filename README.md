# Credit-Card-Fraud-Detection

Data set Link:  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Objective: To Successfully and Accurately identify and flag fraudulent transactions made by a credit card user.

We have used the above-mentioned dataset which contains data collected by a German bank about transactions made on two days in September 2013. 
The dataset contains 31 variables out of which 28 are PCA and mentioned as V1, V2, V3, …, V28. The data is highly imbalanced with the fraud transactions comprising 
only 0.172% of the data. We use the SMOTE technique to over-sample the data and change the weightage of genuine and fraudulent transactions to 65% and 35% respectively.
Then we use the sample() function to decrease the time complexity of the project and randomly select 50000 data values from the SMOTE output. 
We use these 50000 values as the dataset going ahead. We divide the data into training and testing model with the split ratio of 70%. 
We have used the traincontrol function in the caret package to train the Random Forest and Logistic Regression models with the cross-validation method. 
We have used the confusion matrix to evaluate these models.



Random Forest
Accuracy = 99.51%
Sensitivity = 99.82%
Specificity = 98.95%
Precision = 99.81%

Logistic Regression
Accuracy = 96.69%
Sensitivity = 99.18%
Specificity = 92.05%
Precision = 99.18%

Conclusions:  As a result of this experiment, we found that on this dataset with the given 
measures, Random Forest proves to be a better algorithm than Logistic Regression. While the accuracy of both models is close, Random Forest has better Sensitivity and Specificity.
We have used 50000 data points out of the total because as we increase the data points the time complexity of the model increases exponentially with very minute changes in accuracy.
